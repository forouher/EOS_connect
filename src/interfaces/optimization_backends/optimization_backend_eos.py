"""
This module provides the EOSBackend class for interacting with the EOS optimization server.
It includes methods for sending optimization requests, managing configuration, and handling
measurement data.
"""

import logging
import time
import json
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from packaging import version
import pytz

logger = logging.getLogger("__main__")
GENETIC_API_MIN_VERSION = "0.3.0.dev0"


class EOSBackend:
    """
    Backend for direct EOS server optimization.
    Accepts and returns EOS-format requests/responses.
    """

    def __init__(self, base_url, time_frame_base, time_zone):
        self.base_url = base_url
        self.time_frame_base = time_frame_base
        self.time_zone = time_zone
        self.last_optimization_runtimes = [0] * 5
        self.last_optimization_runtime_number = 0
        self._genetic_devices_signature = None
        self.eos_version = "0.0.2"  # default
        try:
            self.eos_version = self._retrieve_eos_version()
            # if self.eos_version in ["0.1.0+dev", "0.2.0+dev"]:
            if self.is_eos_version_at_least("0.1.0"):
                # check config for needed values
                config_optimization = self.__get_config_path("optimization")
                config_optimization_upodate_needed = False
                if self.supports_genetic_api:
                    if config_optimization.get("algorithm") != "GENETIC":
                        config_optimization["algorithm"] = "GENETIC"
                        config_optimization_upodate_needed = True
                    genetic_settings = config_optimization.setdefault("genetic", {})
                    desired_genetic_settings = {
                        "interval_sec": self.time_frame_base,
                        "horizon_hours": 24,
                        "tail_horizon_hours": 0,
                        "individuals": 300,
                        "generations": 400,
                    }
                    for key, value in desired_genetic_settings.items():
                        if genetic_settings.get(key) != value:
                            genetic_settings[key] = value
                            config_optimization_upodate_needed = True
                else:
                    if config_optimization.get("horizon_hours", 0) != 48:
                        config_optimization["horizon_hours"] = 48
                        config_optimization_upodate_needed = True
                    if config_optimization.get("genetic", None) is None:
                        config_optimization["genetic"] = {
                            "individuals": 300,
                            "generations": 400,
                        }
                        config_optimization_upodate_needed = True
                if config_optimization_upodate_needed:
                    self.__set_config_path("optimization", config_optimization)
                    logger.warning(
                        "[OPT-EOS] Detected EOS version >= 0.1.0 - config updated with "
                        + ": %s",
                        config_optimization,
                    )
                else:
                    logger.info(
                        "[OPT-EOS] Detected EOS version >= 0.1.0 - config optimization values OK"
                    )

                config_devices = self.__get_config_path("devices/electric_vehicles")
                if not self.supports_genetic_api:
                    if config_devices is None:
                        config_devices = [{}]
                        config_devices[0]["charge_rates"] = sorted(
                            [0.0, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
                        )
                        self.__set_config_path(
                            "devices/electric_vehicles", config_devices
                        )
                        logger.warning("[OPT-EOS] Legacy EV charge rates configured")
                    elif "charge_rates" not in config_devices[0]:
                        config_devices[0]["charge_rates"] = sorted(
                            [0.0, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
                        )
                        self.__set_config_path(
                            "devices/electric_vehicles", config_devices
                        )
                        logger.warning("[OPT-EOS] Legacy EV charge rates configured")
                    else:
                        logger.info("[OPT-EOS] Legacy EV charge rates are configured")
            logger.info("[OPT-EOS] Configuration validation successful")
        except ValueError as e:
            logger.error("[OPT-EOS] EOS backend configuration error: %s", str(e))
            logger.error(
                "[OPT-EOS] EOS backend will start in degraded mode - "
                "Use Settings to fix configuration"
            )
            raise  # Let caller handle graceful degradation instead of sys.exit(1)

    @property
    def supports_genetic_api(self):
        """Whether the connected EOS server exposes the configuration-owned GENETIC API."""
        return self.is_eos_version_at_least(GENETIC_API_MIN_VERSION)

    def _get_expected_hourly_slots(self):
        """
        Returns the expected number of hourly slots for the 2-day window
        (today midnight to day-after-tomorrow midnight) that the EOS server
        validates arrays against.  Handles DST:
        - Normal day   : 48
        - Spring-forward: 47 (today = 23 h)
        - Fall-back     : 49 (today = 25 h)
        """
        tz = self.time_zone
        if isinstance(tz, str):
            tz = pytz.timezone(tz)
        today = datetime.now(tz).date()
        today_midnight = tz.localize(
            datetime(today.year, today.month, today.day, 0, 0, 0)
        )
        day_after = today + timedelta(days=2)
        day_after_midnight = tz.localize(
            datetime(day_after.year, day_after.month, day_after.day, 0, 0, 0)
        )
        return int((day_after_midnight - today_midnight).total_seconds()) // 3600

    def _adjust_arrays_for_dst(self, eos_request, expected_hourly):
        """
        Trim or pad all time-series arrays inside eos_request so every
        array has exactly *expected_hourly* elements (or expected_hourly * 4
        for 15-min mode).  Sources always produce 48/192 slots; on DST days
        old EOS servers (< 0.1.0) computed a DST-aware required slot count
        (47 on spring-forward, 49 on fall-back) and rejected mismatched arrays.

        Modern EOS builds use the forecast duration (normally 48 slots) and
        handle DST internally. Trimming or padding the arrays for those
        versions causes the optimizer to receive incomplete data, so the
        adjustment remains limited to legacy servers.

        The adjustment is done in-place on eos_request.
        """
        if expected_hourly == 48:
            return  # Normal day – nothing to do

        # Modern EOS versions handle DST internally and expect the normal
        # forecast duration, rather than a calendar-day-adjusted array.
        if self.is_eos_version_at_least("0.1.0"):
            logger.debug(
                "[OPT-EOS] DST day (%d wall-clock hours) – skipping array adjustment "
                "because modern EOS handles DST internally. Sending the standard "
                "forecast arrays.",
                expected_hourly,
            )
            return

        expected_slots = (
            expected_hourly * 4 if self.time_frame_base == 900 else expected_hourly
        )

        def _fit(arr, fill_val=0.0):
            """Return arr trimmed or padded to expected_slots."""
            if not arr:
                return arr
            if len(arr) > expected_slots:
                return arr[:expected_slots]
            if len(arr) < expected_slots:
                pad = arr[-1] if arr else fill_val
                return arr + [pad] * (expected_slots - len(arr))
            return arr

        direction = "trimmed" if expected_hourly < 48 else "extended"
        logger.info(
            "[OPT-EOS] DST day detected: expected %d hourly slots (%d total slots). "
            "Arrays will be %s from standard 48-slot baseline.",
            expected_hourly,
            expected_slots,
            direction,
        )

        ems = eos_request.get("ems", {})
        for key in (
            "pv_prognose_wh",
            "strompreis_euro_pro_wh",
            "einspeiseverguetung_euro_pro_wh",
            "gesamtlast",
        ):
            if key in ems:
                ems[key] = _fit(ems[key])

        if "temperature_forecast" in eos_request:
            eos_request["temperature_forecast"] = _fit(
                eos_request["temperature_forecast"]
            )

    def optimize(self, eos_request, timeout=180):
        """
        Send the optimize request to the EOS server.
        Returns (response_json, avg_runtime)
        """
        if self.supports_genetic_api:
            # Modern EOS handles DST internally, so _adjust_arrays_for_dst is a
            # no-op for the GENETIC endpoint.
            expected_hourly = self._get_expected_hourly_slots()
            self._adjust_arrays_for_dst(eos_request, expected_hourly)
            self._sync_genetic_devices(eos_request)
            request_payload = self._to_genetic_request(eos_request)
            request_url = self.base_url + "/v1/optimize"
        else:
            # Older EOS versions only expose the legacy GENETIC0 endpoint.
            expected_hourly = self._get_expected_hourly_slots()
            self._adjust_arrays_for_dst(eos_request, expected_hourly)
            request_payload = eos_request
            request_url = (
                self.base_url
                + "/optimize"
                + "?start_hour="
                + str(datetime.now(self.time_zone).hour)
            )

        headers = {"accept": "application/json", "Content-Type": "application/json"}
        logger.info(
            "[OPT-EOS] OPTIMIZE request optimization with: %s - and with timeout: %s",
            request_url,
            timeout,
        )
        response = None
        try:
            start_time = time.time()
            response = requests.post(
                request_url, headers=headers, json=request_payload, timeout=timeout
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            logger.info(
                "[OPT-EOS] OPTIMIZE response retrieved successfully "
                "in %d min %.2f sec for current run",
                int(minutes),
                seconds,
            )
            response.raise_for_status()
            # Store runtime in circular list
            if all(runtime == 0 for runtime in self.last_optimization_runtimes):
                self.last_optimization_runtimes = [elapsed_time] * 5
            else:
                self.last_optimization_runtimes[
                    self.last_optimization_runtime_number
                ] = elapsed_time
            self.last_optimization_runtime_number = (
                self.last_optimization_runtime_number + 1
            ) % 5
            avg_runtime = sum(self.last_optimization_runtimes) / 5
            response_json = response.json()
            if self.supports_genetic_api:
                response_json = self._normalize_genetic_response(response_json)
            return response_json, avg_runtime
        except requests.exceptions.Timeout:
            logger.error(
                "[OPT-EOS] OPTIMIZE Request timed out after %s seconds | "
                "Config: #eos | ACTION REQUIRED",
                timeout,
            )
            return {"error": "Request timed out - trying again with next run"}, None
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "[OPT-EOS] OPTIMIZE Connection error - EOS server not reachable at %s "
                "will try again with next cycle - error: %s | Config: #eos | ACTION REQUIRED",
                request_url,
                str(e),
            )
            return {
                "error": f"EOS server not reachable at {self.base_url} "
                "will try again with next cycle"
            }, None
        except requests.exceptions.RequestException as e:
            logger.error("[OPT-EOS] OPTIMIZE Request failed: %s", e)
            if response is not None:
                logger.error(
                    "[OPT-EOS] OPTIMIZE Response status: %s", response.status_code
                )
                logger.debug(
                    "[OPT-EOS] OPTIMIZE ERROR - response of EOS is:\n%s",
                    response.text,
                )
            logger.debug(
                "[OPT-EOS] OPTIMIZE ERROR - payload for the request was:\n%s",
                eos_request,
            )
            # The exception text is logged above but not returned: it carries the
            # resolved host, port and any proxy in the chain, and this dict is served
            # verbatim by GET /json/optimize_response.json.
            return {
                "error": f"Request to the EOS server at {self.base_url} failed "
                "- see the log for details, will try again with next cycle"
            }, None

    def _to_genetic_request(self, eos_request):
        """Translate the legacy connector payload to the GENETIC request format."""
        ems = eos_request.get("ems", {})
        forecasts = {
            "pv_forecast_wh": ems.get("pv_prognose_wh"),
            "electricity_price_per_wh": ems.get("strompreis_euro_pro_wh"),
            "feed_in_tariff_per_wh": ems.get("einspeiseverguetung_euro_pro_wh"),
            "total_load": ems.get("gesamtlast"),
        }
        if eos_request.get("temperature_forecast") is not None:
            forecasts["temperature_forecast"] = eos_request["temperature_forecast"]

        soc = {}
        for key in ("pv_akku", "eauto"):
            device = eos_request.get(key) or {}
            if (
                device.get("device_id")
                and device.get("initial_soc_percentage") is not None
            ):
                soc[device["device_id"]] = device["initial_soc_percentage"]

        request = {"forecasts": forecasts}
        if soc:
            request["soc"] = soc
        if eos_request.get("start_solution") is not None:
            request["start_solution"] = eos_request["start_solution"]
        return request

    def _sync_genetic_devices(self, eos_request):
        """Keep the remote GENETIC device configuration aligned with the connector."""
        battery = dict(eos_request.get("pv_akku") or {})
        if battery:
            battery["device_id"] = battery.get("device_id", "battery1")
        ev = dict(eos_request.get("eauto") or {})
        if ev:
            ev["device_id"] = ev.get("device_id", "ev1")
        inverter = dict(eos_request.get("inverter") or {})
        if inverter:
            inverter["device_id"] = inverter.get("device_id", "inverter1")
            if "max_power_wh" in inverter:
                inverter["max_power_w"] = inverter.pop("max_power_wh")
        appliance = dict(eos_request.get("dishwasher") or {})
        if appliance:
            appliance["device_id"] = appliance.get("device_id", "additional_load_1")
            appliance.setdefault("num_cycles", 1)

        updates = {
            "devices/batteries": {battery["device_id"]: battery} if battery else {},
            "devices/electric_vehicles": {ev["device_id"]: ev} if ev else {},
            "devices/inverters": {inverter["device_id"]: inverter} if inverter else {},
            "devices/home_appliances": (
                {appliance["device_id"]: appliance} if appliance else {}
            ),
            "devices/max_batteries": 1 if battery else 0,
            "devices/max_electric_vehicles": 1 if ev else 0,
            "devices/max_inverters": 1 if inverter else 0,
            "devices/max_home_appliances": 1 if appliance else 0,
        }
        signature = json.dumps(updates, sort_keys=True, default=str)
        if signature == self._genetic_devices_signature:
            return
        for path, value in updates.items():
            self.__set_config_path(path, value)
        self._genetic_devices_signature = signature

    def _normalize_genetic_response(self, response):
        """Adapt GENETIC's run-relative controls to EOS Connect's midnight grid."""
        if not isinstance(response, dict):
            return response

        if response.get("controls_start_at_now"):
            now = datetime.now(self.time_zone)
            midnight = self.time_zone.localize(
                datetime(now.year, now.month, now.day, 0, 0, 0)
            )
            current_slot = int((now - midnight).total_seconds()) // self.time_frame_base
            for key in ("ac_charge", "dc_charge", "discharge_allowed"):
                if key in response:
                    response[key] = [0] * current_slot + response[key]

        if "washingstart" not in response:
            starts = response.get("appliance_starts") or {}
            first_start = next(iter(starts.values()), [])
            if first_start:
                timestamp = first_start[0]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone(self.time_zone)
                else:
                    timestamp = self.time_zone.localize(timestamp)
                now = datetime.now(self.time_zone)
                midnight = self.time_zone.localize(
                    datetime(now.year, now.month, now.day, 0, 0, 0)
                )
                response["washingstart"] = int(
                    (timestamp - midnight).total_seconds() // self.time_frame_base
                )
        return response

    def __get_config_path(self, path):
        """
        Get a configuration value from the EOS server.
        """
        # Always specify a timeout to avoid hanging indefinitely
        response = requests.get(self.base_url + "/v1/config/" + path, timeout=10)
        response.raise_for_status()
        config_value = response.json()
        return config_value

    def __set_config_path(self, path, value):
        """
        Set a configuration value on the EOS server.

        Args:
            path (str): The configuration path.
            value (dict or list): The configuration value as a JSON-serializable
            object or a list of such objects.
        """

        def convert_sets(obj):
            """Recursively convert sets to lists in a dict or list."""
            if isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(v) for v in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj

        headers = {"Content-Type": "application/json"}
        try:
            value_serializable = convert_sets(value)
            response = requests.put(
                self.base_url + "/v1/config/" + path,
                data=json.dumps(value_serializable),
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(
                "[OPT-EOS] Failed to set config value for path '%s': %s", path, e
            )

    def send_measurement_to_eos(self, dataframe):
        """
        Send the measurement data to the EOS server.
        """
        params = {
            "data": dataframe.to_json(orient="index"),
            "dtype": "float64",
            "tz": "UTC",
        }
        response = requests.put(
            self.base_url
            + "/v1/measurement/load-mr/series/by-name"
            + "?name=Household",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        if response.status_code == 200:
            logger.debug("[OPT-EOS] Measurement data sent to EOS server successfully.")
        else:
            logger.debug(
                "[OPT-EOS] Failed to send data to EOS server. Status code: %s, Response: %s",
                response.status_code,
                response.text,
            )

    def save_config_to_config_file(self):
        """
        Save the current configuration to the configuration file on the EOS server.
        """
        response = requests.put(self.base_url + "/v1/config/file", timeout=10)
        response.raise_for_status()
        logger.debug("[OPT-EOS] CONFIG saved to config file successfully.")

    def update_config_from_config_file(self):
        """
        Update the current configuration from the configuration file on the EOS server.
        """
        try:
            response = requests.post(self.base_url + "/v1/config/update", timeout=10)
            response.raise_for_status()
            logger.info("[OPT-EOS] CONFIG updated from config file successfully.")
        except requests.exceptions.Timeout:
            logger.error(
                "[OPT-EOS] CONFIG Request timed out while updating config from config file."
            )
        except requests.exceptions.RequestException as e:
            logger.error(
                "[OPT-EOS] CONFIG Request failed while updating config from config file: %s",
                e,
            )

    def _retrieve_eos_version(self):
        """
        Get the EOS version from the server.
        Returns: str
        """
        try:
            response = requests.get(self.base_url + "/v1/health", timeout=10)
            response.raise_for_status()
            eos_version = response.json().get("status")
            eos_version_real = response.json().get("version", "unknown")
            if eos_version == "alive" and eos_version_real == "unknown":
                eos_version = "0.0.2"
            else:
                eos_version = eos_version_real
                # raise ValueError(
                #     f"EOS version {eos_version_real} currently not supported!"
                # )
            logger.info("[OPT-EOS] Getting EOS version: %s", eos_version)
            return eos_version
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response and e.response.status_code == 404:
                eos_version = "0.0.1"
                logger.info("[OPT-EOS] Getting EOS version: %s", eos_version)
                return eos_version
            else:
                logger.error(
                    "[OPT-EOS] HTTP error occurred while getting EOS version - use preset version:"
                    + " %s : %s - Response: %s",
                    self.eos_version,
                    e,
                    (
                        e.response.text
                        if hasattr(e, "response") and e.response
                        else "No response"
                    ),
                )
                return self.eos_version
        except requests.exceptions.ConnectTimeout:
            logger.error(
                "[OPT-EOS] Failed to get EOS version  - use preset version: '%s' - Server not "
                + "reachable: Connection to %s timed out | Config: #eos | ACTION REQUIRED",
                self.eos_version,
                self.base_url,
            )
            return self.eos_version
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "[OPT-EOS] Failed to get EOS version - use preset version: '%s' - "
                "Connection error: %s | Config: #eos | ACTION REQUIRED",
                self.eos_version,
                e,
            )
            return self.eos_version
        except requests.exceptions.RequestException as e:
            logger.error(
                "[OPT-EOS] Failed to get EOS version - use preset version: '%s' - "
                "Error: %s | Config: #eos | ACTION REQUIRED",
                self.eos_version,
                e,
            )
            return self.eos_version
        except json.JSONDecodeError as e:
            logger.error(
                "[OPT-EOS] Failed to decode EOS version - use preset version: '%s' - response: %s ",
                self.eos_version,
                e,
            )
            return self.eos_version

    def get_eos_version(self):
        """
        Get the EOS version from the server.
        Returns: str
        """
        return self.eos_version

    def is_eos_version_at_least(self, version_string):
        """
        Check if the EOS version is at least the given version.
        Args:
            version_string (str): Version string to compare against.
        Returns: bool
        """
        try:
            return version.parse(self.eos_version) >= version.parse(version_string)
        except ImportError:
            logger.warning(
                "[OPT-EOS] 'packaging' module not found. Cannot compare EOS versions."
            )
            return False
        except version.InvalidVersion:
            logger.warning(
                "[OPT-EOS] Invalid EOS version string '%s'. "
                "Cannot compare versions. Assuming version < %s",
                self.eos_version,
                version_string,
            )
            return False

    def create_dataframe(self, profile):
        """
        Creates a pandas DataFrame with hourly energy values for a given profile.
        Args:
            profile (list of tuples): Each tuple: (month, weekday, hour, energy)
        Returns:
            pandas.DataFrame: DateTime index for 2025, 'Household' column.
        """
        dates = pd.date_range(start="1/1/2025", end="31/12/2025", freq="H")
        df = pd.DataFrame(index=dates)
        df["Household"] = np.nan
        for entry in profile:
            month, weekday, hour, energy = entry
            matching_dates = df[
                (df.index.month == month)
                & (df.index.weekday == weekday)
                & (df.index.hour == hour)
            ].index
            for date in matching_dates:
                df.loc[date, "Household"] = energy
        return df

    def _validate_eos_input(self, eos_request):
        """
        Validate EOS-format optimization request.
        Returns: (bool, list[str]) - valid, errors
        """
        errors = []
        if not isinstance(eos_request, dict):
            errors.append("Request must be a dictionary.")
        # Add more checks as needed
        # Example: check required keys
        required_keys = ["ems", "pv_akku"]
        for key in required_keys:
            if key not in eos_request:
                errors.append(f"Missing required key: {key}")
        return len(errors) == 0, errors

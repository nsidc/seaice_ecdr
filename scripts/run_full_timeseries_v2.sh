#!/bin/bash

# G10016
ecdr daily-nrt --start-date 2025-08-01 --end-date 2025-09-29 --hemisphere both

ecdr monthly-nrt --year 2025 --month 08 --hemisphere both

# G02202
ecdr daily --start-date 1978-10-25 --end-date 2025-09-23 --hemisphere both
ecdr daily --start-date 1978-10-25 --end-date 2025-09-23 --hemisphere both

ecdr monthly --year 1978 --month 11 --end-year 2025 --end-month 8 --hemisphere both

ecdr daily-aggregate --year 1978 --end-year 2024 --hemisphere north
ecdr daily-aggregate --year 1978 --end-year 2024 --hemisphere south

ecdr monthly-aggregate --hemisphere north
ecdr monthly-aggregate --hemisphere south

ecdr validate-outputs --hemisphere both --start-date 1978-10-25 --end-date 2025-09-23

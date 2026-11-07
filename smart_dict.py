# smart_dict.py - Comprehensive S.M.A.R.T. Attributes Dictionary

SMART_DEFINITIONS = {
    "1": {
        "name": "S.M.A.R.T. 1 - Read Error Rate",
        "desc": "Stores hardware-level read error rate while reading data from the disk surface.",
        "cause": "Damaged magnetic platter surface or degraded read/head assembly."
    },
    "3": {
        "name": "S.M.A.R.T. 3 - Spin-Up Time",
        "desc": "Average time interval required for spindle spin-up from zero to operating RPM.",
        "cause": "Spindle motor bearing wear or electrical motor friction."
    },
    "4": {
        "name": "S.M.A.R.T. 4 - Start/Stop Count",
        "desc": "Cumulative count of spindle start/stop cycles over the drive's operational lifetime.",
        "cause": "Frequent power cycling, aggressive OS sleep states, or normal operational aging."
    },
    "5": {
        "name": "S.M.A.R.T. 5 - Reallocated Sectors Count",
        "desc": "Count of bad physical sectors found and remapped to spare reserve tracks.",
        "cause": "Surface media degradation, mechanical friction, or physical shock/vibration."
    },
    "9": {
        "name": "S.M.A.R.T. 9 - Power-On Hours (POH)",
        "desc": "Cumulative total time the drive has been powered on during its operational lifecycle.",
        "cause": "Normal component aging, bearing wear, and thermal expansion over time."
    },
    "10": {
        "name": "S.M.A.R.T. 10 - Spin Retry Count",
        "desc": "Count of spin-up attempts required to reach target operating speed if the first attempt failed.",
        "cause": "Spindle motor fatigue or electrical power delivery fluctuations."
    },
    "12": {
        "name": "S.M.A.R.T. 12 - Power Cycle Count",
        "desc": "Total number of complete power on/off cycles experienced by the device.",
        "cause": "System reboots, hard shutdowns, or power supply toggling."
    },
    "183": {
        "name": "S.M.A.R.T. 183 - Runtime Bad Block",
        "desc": "Number of runtime bad blocks detected during data operations (common in SSDs).",
        "cause": "NAND flash wear-out, block allocation errors, or cell degradation."
    },
    "184": {
        "name": "S.M.A.R.T. 184 - End-to-End Error",
        "desc": "Part of S.M.A.R.T. parity error tracking; data transferred between host and cache RAM had parity mismatch.",
        "cause": "Faulty RAM cache controller chips, poor cabling, or firmware bugs."
    },
    "187": {
        "name": "S.M.A.R.T. 187 - Reported Uncorrectable Errors",
        "desc": "Count of read/write errors that could not be recovered using hardware error correction codes (ECC).",
        "cause": "Severe platter surface damage, head degradation, or uncorrectable magnetic decay."
    },
    "188": {
        "name": "S.M.A.R.T. 188 - Command Timeout",
        "desc": "Count of aborted operations due to drive controller response timeouts.",
        "cause": "Power delivery voltage instability, faulty SATA/NVMe cables, or controller firmware lockups."
    },
    "190": {
        "name": "S.M.A.R.T. 190 - Airflow Temperature / Temperature",
        "desc": "Internal drive operating temperature sensor tracking thermal stability.",
        "cause": "Inadequate chassis cooling, high ambient room temperature, or heavy sustained workloads."
    },
    "193": {
        "name": "S.M.A.R.T. 193 - Load/Unload Cycle Count",
        "desc": "Total count of head load/unload operations moving onto or off the magnetic platter surface.",
        "cause": "Frequent aggressive head parking, power management policies, or idle timeouts."
    },
    "194": {
        "name": "S.M.A.R.T. 194 - Temperature",
        "desc": "Current internal temperature measured by the device thermal sensor.",
        "cause": "Thermal dissipation issues or restricted airflow within the PC enclosure."
    },
    "196": {
        "name": "S.M.A.R.T. 196 - Reallocation Event Count",
        "desc": "Total count of completed and attempted mapping operations for unstable sectors.",
        "cause": "Ongoing platter degradation and expanding bad sector clusters."
    },
    "197": {
        "name": "S.M.A.R.T. 197 - Current Pending Sector Count",
        "desc": "Unstable sectors waiting to be remapped due to read/write verification errors.",
        "cause": "Weak magnetic signals or pending sector instability awaiting successful reallocation."
    },
    "198": {
        "name": "S.M.A.R.T. 198 - Offline Uncorrectable Sector Count",
        "desc": "Sectors that are completely inaccessible and could not be verified or corrected during self-tests.",
        "cause": "Advanced mechanical failure or deep physical damage to the magnetic disk medium."
    },
    "199": {
        "name": "S.M.A.R.T. 199 - UDMA CRC Error Count",
        "desc": "Number of errors encountered communicating via the interface cable (CRC checksum failures).",
        "cause": "Faulty, loose, or unshielded SATA/SAS data cables, or failing motherboard ports."
    }
}
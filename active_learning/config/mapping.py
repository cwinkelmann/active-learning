prefix_mapping = {
    "Española": "Esp_E",
    "Española": "Esp_E",
    "Espanola": "Esp_E",
    "Fernandina": "Fer_F",
    "Floreana": "Flo_FL",
    "Genovesa": "Gen_G",
    "Isabela": "Isa_IS",
    "Marchena": "Mar_M",
    "Pinta": "Pin_P",
    "Pinzón": "Pnz_PZ",
    "Rábida": "Rab_RA",
    "San Cristóbal": "Scris_SR",
    "Santa Cruz": "Scruz_SC",
    "Santa Fe": "SanFe_SF",
    "Santiago": "Snt_ST",
    "Wolf": "Wol_W",
    "Bartolomé (Santiago)": "Snt_BT", # TODO they are not matching the filenames/folder names
    "Bainbridge (Santiago)": "Snt_BA",
    "Beagles (Santiago)": "Snt_BE",
    "Daphnes (Santa Cruz)": "Scruz_DA",
    "Plazas (Santa Cruz)": "Scruz_PL",
    "Seymour (Santa Cruz)": "Scruz_SE",
}


def get_island_code():
    """
    Create a mapping dictionary that extracts the part before '_' from each value
    while preserving the original keys.

    Parameters:
    original_codes (dict): Original dictionary with island codes as values

    Returns:
    dict: New dictionary with same keys but shortened values (part before '_')
    """
    mapping = {}

    for key, value in prefix_mapping.items():
        # Extract the part before '_' if there is one, otherwise keep the original value
        if isinstance(value, str) and '_' in value:
            mapping[key] = value.split('_')[0]
        else:
            mapping[key] = value

    return mapping

drone_mapping = {
    '0K8TFAK0025193': 'Sula',
    '0K8TH7Q0121445': 'Nazca',
    '0K8TG920120534': 'Condor',
    '0K8THB60120896': 'Mockingbird',
    '0K8TG140020164': 'Owl',
    '0K8TH9C0120310': 'Brujo',
    '0K8TH7F0120402': 'Hawk',
    '0K8THC20120846': 'Unknown_2',
    '0K8THA50120194': 'Unknown_3',
}

# TODO this has to become more flexible
keypoint_id_mapping = {
    "iguana_point" : "ed18e0f9-095f-46ff-bc95-febf4a53f0ff",
    "iguana" : "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"

}
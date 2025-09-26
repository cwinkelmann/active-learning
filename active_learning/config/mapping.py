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



expedition_mapping = {
    "2020_01": 1,
    "2021_01": 2,
    "2021_02": 2,
    "2021_12": 3,
    "2023_01": 4,
    "2023_02": 4,
    "2024_01": 5,  # from the El Niño folder
    "2024_04": 6,  # from the El Niño folder
    "2024_05": 6,  # from the El Niño folder
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
        '0K8THC20120846': 'Tropical bird',
        '0K8THA50120194': 'Pelicano',
    # The mavic Enterprise woud be Albatross
}

# TODO this has to become more flexible
keypoint_id_mapping = {
    "iguana_point" : "ed18e0f9-095f-46ff-bc95-febf4a53f0ff",
    "iguana" : "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"

}

mission_names_filter = [
    # Floreana, clockwise order
    {"island_name": "Floreana", "missions": ["FLMO04_03022021", "FLMO05_03022021", "FLMO06_03022021"]},
    {"island_name": "Floreana", "missions": ["FLMO01_02022021", "FLMO02_02022021", "FLMO03_02022021"]},
    {"island_name": "Floreana", "missions": ["FLMO02_28012023"]},
    {"island_name": "Floreana", "missions": ["FLBB01_28012023"]},  # intersection with annotated raster
    {"island_name": "Floreana", "missions": ["FLPC07_22012021"]},  # intersection with annotated raster
    {"island_name": "Floreana", "missions": ["FLPA03_21012021"]},
    {"island_name": "Floreana", "missions": ["FLSCA02_23012021"]},

    # Genovesa, clockwise order
    {"island_name": "Genovesa", "missions": ["GES06_04122021", "GES07_04122021"]},  # intersection with annotated raster
    {"island_name": "Genovesa", "missions": ["GES13_05122021"]},  # intersection with annotated raster

    # Santiago
    {"island_name": "Santiago", "missions": ["STJB01_10012023"]},

    # Fernandina
    {"island_name": "Fernandina", "missions": ["FCD01_20122021", "FCD02_20122021", "FCD03_20122021"]},
    {"island_name": "Fernandina", "missions": ["FPE01_18122021"]},
    {"island_name": "Fernandina", "missions": ["FEA01_18122021"]},  # accidentally assigned to Floreana in Hasty
]


orthomosaic_training_mapping = [
    {
        "experiment": "HQ_body",
        "orthomosaic": "FLPC01-07_22012021",
        "programm": "Pix4D", "path": "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff",
        "annotations": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/FLPC01_07_22012021.gpkg"
    },
    {
        "experiment": "LQ_body",
        "orthomosaic": "FLPC01-07_22012021",
        "programm": "Metashape",
        "path": "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff",
        "annotations_gpkg": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/FLPC01_07_22012021.gpkg",
        "layer_name": "Flo_FLPC01_22012021 counts"
    },
    {
        "experiment": "LQ_body",
        "orthomosaic": "FLPC01-07_22012021",
        "programm": "Metashape",
        "path": "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff",
        "annotations_gpkg": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/FLPC01_07_22012021.gpkg",
        "layer_name": "Flo_FLPC01_22012021 counts"
    }



]
import dataclasses
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict

import khandy


@dataclass
class Class:
    chinese_name: str
    latin_name: str


@dataclass
class Superclass:
    latin_name: str
    class_indices: List[int]


@dataclass
class Labelmap:
    species_taxons: Dict[int, Class]
    genus_taxons: Dict[str, Superclass]
    family_taxons: Dict[str, Superclass]


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '../plantid/models/')
    label_map_path = os.path.join(model_dir, 'quarrying_plantid_label_map.txt')
    family_name_map_path = os.path.join(model_dir, 'family_name_map.json')
    genus_name_map_path = os.path.join(model_dir, 'genus_name_map.json')

    family_name_map = khandy.load_json(family_name_map_path)
    genus_name_map = khandy.load_json(genus_name_map_path)

    records = khandy.load_list(label_map_path)
    species_taxons = {}
    genus_to_class_indices = {}
    family_to_class_indices = {}
    for record in records:
        label, chinese_name, latin_name = record.split(',')
        species_taxons[int(label)] = Class(chinese_name, latin_name)

        # genus = Superclass(latin)
        underscore_parts = chinese_name.split('_')
        if len(underscore_parts) == 1:
            family_to_class_indices.setdefault(underscore_parts[0], []).append(int(label))
            genus_to_class_indices.setdefault(underscore_parts[0], []).append(int(label))
        elif len(underscore_parts) > 1:
            family_to_class_indices.setdefault(underscore_parts[0], []).append(int(label))
            genus_to_class_indices.setdefault('_'.join(underscore_parts[:2]), []).append(int(label))

    family_taxons = {}
    for name, class_indices in family_to_class_indices.items():
        family_latin_name = family_name_map.get(name, '')
        family_taxons[name] = Superclass(family_latin_name, class_indices)

    genus_taxons = {}
    for name, class_indices in genus_to_class_indices.items():
        genus_latin_name = genus_name_map.get(name, '')
        genus_taxons[name] = Superclass(genus_latin_name, class_indices)

    labelmap = Labelmap(species_taxons, genus_taxons, family_taxons)
    khandy.save_json(os.path.join(model_dir, 'quarrying_plantid_label_map.json'), dataclasses.asdict(labelmap))


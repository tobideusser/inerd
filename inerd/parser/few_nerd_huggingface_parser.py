from typing import Optional

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from inerd.data_classes import Sentence, NERCorpus
from inerd.parser import BaseParser


class FewNERDHuggingFaceParser(BaseParser):
    def __init__(
        self,
        entity_separator_token: str,
        type_content_separator_token: str,
        type_mapping: bool = True,
        debug_size: Optional[int] = None,
        dataset_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if type_mapping:
            type_mapping = {
                "art-broadcastprogram": "Broadcast program",
                "art-film": "Film",
                "art-music": "Music",
                "art-other": "Other art",
                "art-painting": "Painting",
                "art-writtenart": "Written art",
                "building-airport": "Airport",
                "building-hospita": "Hospital",
                "building-hotel": "Hotel",
                "building-library": "Library",
                "building-other": "Other building",
                "building-restaurant": "Restaurant",
                "building-sportsfacility": "Sports facility",
                "building-theater": "Theater",
                "event-attack/battle/war/militaryconflict": "Military conflict",
                "event-disaster": "Disaster",
                "event-election": "Election",
                "event-other": "Other event",
                "event-protest": "Protest",
                "event-sportsevent": "Sports event",
                "location-GPE": "Geopolitical",
                "location-bodiesofwater": "Body of water",
                "location-island": "Island",
                "location-mountain": "Mountain",
                "location-other": "Other location",
                "location-park": "Park",
                "location-road/railway/highway/transit": "Transportation infrastructure",
                "organization-company": "Company",
                "organization-education": "Educational institutions",
                "organization-government/governmentagency": "Government",
                "organization-media/newspaper": "media",
                "organization-other": "Other organization",
                "organization-politicalparty": "Political party",
                "organization-religion": "Religion",
                "organization-showorganization": "Show",
                "organization-sportsleague": "Sports league",
                "organization-sportsteam": "Sports team",
                "other-astronomything": "Astronomy",
                "other-award": "Award",
                "other-biologything": "Biology",
                "other-chemicalthing": "Chemical",
                "other-currency": "Currency",
                "other-disease": "Disease",
                "other-educationaldegree": "Educational degree",
                "other-god": "God",
                "other-language": "Language",
                "other-law": "Law",
                "other-livingthing": "Living thing",
                "other-medical": "Medical",
                "person-actor": "Actor",
                "person-artist/author": "Artist",
                "person-athlete": "Athlete",
                "person-director": "Director",
                "person-other": "Other person",
                "person-politician": "Politician",
                "person-scholar": "Scholar",
                "person-soldier": "Soldier",
                "product-airplane": "Airplane",
                "product-car": "Car",
                "product-food": "Food",
                "product-game": "Game",
                "product-other": "Other product",
                "product-ship": "Ship",
                "product-software": "Software",
                "product-train": "Train",
                "product-weapon": "Weapon",
            }
        else:
            type_mapping = None
        super().__init__(
            type_mapping=type_mapping,
            debug_size=debug_size,
            entity_separator_token=entity_separator_token,
            type_content_separator_token=type_content_separator_token,
            cache_dir=cache_dir,
        )
        self.dataset_name = dataset_name if dataset_name else "FewNERD"

    def _parse_few_nerd_split(self, dataset: DatasetDict, split_type: str):
        parsed = []
        i = 0
        for sentence in tqdm(
            dataset[split_type],
            desc=f"Parsing {split_type}",
            total=len(dataset[split_type]),
        ):
            entities = self._entity_tags_to_entity_dict(
                entity_tags=sentence["fine_ner_tags"], words=sentence["tokens"], tagging_type="simple"
            )
            parsed.append(
                Sentence(
                    id_=split_type + "-" + sentence["id"],
                    words=sentence["tokens"],
                    entity_tags=sentence["fine_ner_tags"],
                    entity_label=[self.entity_tag_to_label[entity_tag] for entity_tag in sentence["fine_ner_tags"]],
                    entities_anno=entities,
                    entity_separator_token=self.entity_separator_token,
                    type_content_separator_token=self.type_content_separator_token,
                )
            )
            i += 1
            if self.debug_size and i >= self.debug_size:
                return parsed
        return parsed

    def parse(self) -> NERCorpus:
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised", cache_dir=self.cache_dir)
        corpus = {"train": [], "validation": [], "test": []}
        entity_labels = dataset["train"].features["fine_ner_tags"].feature.names
        self.entity_tag_to_label = {i: entity_label for i, entity_label in enumerate(entity_labels)}
        if self.type_mapping is not None:
            for k, v in self.entity_tag_to_label.items():
                for kk, vv in self.type_mapping.items():
                    if kk in v:
                        self.entity_tag_to_label[k] = self.entity_tag_to_label[k].replace(kk, vv)
        for split_type in ["train", "validation", "test"]:
            corpus[split_type] = self._parse_few_nerd_split(dataset=dataset, split_type=split_type)

        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["validation"],
            test=corpus["test"],
            name=self.dataset_name,
        )
        return corpus_parsed

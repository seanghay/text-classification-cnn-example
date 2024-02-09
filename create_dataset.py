import csv
import unicodedata
from pprint import pprint
import json

labels = sorted(
    [
        "6228820fdf9b2649a2ad55da",
        "6228820fdf9b2649a2ad55d7",
        "6228820fdf9b2649a2ad55d9",
        "6228820fdf9b2649a2ad55c6",
        "6228820fdf9b2649a2ad55d4",
        "62bd1a3a6132518058eb4cb1",
        "6228820fdf9b2649a2ad55b1",
        "6228820fdf9b2649a2ad55c9",
        "6228820fdf9b2649a2ad55ad",
        "623d8a31ecc9601ec2f4dc83",
    ]
)




if __name__ == "__main__":
    categories = []
    with open("./data/categories.jsonl") as infile:
        for line in infile:
            data = json.loads(line.rstrip("\n"))
            id = data["_id"]["$oid"]
            name = data["name"]
            categories.append((id, name.replace("\u200b", "")))

    categories_dict = dict(categories)
    places = [] 
    idx2labels = { i: categories_dict[label_id] for i, label_id in enumerate(labels) }
    print(idx2labels)
    exit()
    
    with open("./data/result.tsv", "w") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["text", "label_idx", "label_name"])
        with open("./data/places.jsonl") as infile:
            for line in infile:
                line = line.rstrip("\n")
                data = json.loads(line)
                id = data["_id"]["$oid"]
                name = data["name"].replace("\u200b", "")
                categories = list(map(lambda x: x["$oid"], data["categories"]))
                places.append((name, categories))

                for label in categories:
                    if label not in labels:
                      continue
                    
                    writer.writerow(
                        [unicodedata.normalize("NFKC", name.lower()), labels.index(label), categories_dict[label]]
                    )

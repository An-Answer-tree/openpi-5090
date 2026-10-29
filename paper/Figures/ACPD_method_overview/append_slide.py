"""Append a one-slide PptxGenJS deck while preserving the original slide parts."""

import argparse
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("p", P)
ET.register_namespace("r", R)
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")


def xml_bytes(element: ET.Element) -> bytes:
    """Serialize modified package metadata, leaving slide XML untouched."""
    namespace = element.tag[1:].split("}")[0]
    if namespace != P:
        ET.register_namespace("", namespace)
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def main() -> None:
    """Merge a new slide and its layout/master/media into an existing deck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("new_slide", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.source) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(args.new_slide) as archive:
        generated = {name: archive.read(name) for name in archive.namelist()}
    presentation = ET.fromstring(original["ppt/presentation.xml"])
    slide_ids = presentation.find(f"{{{P}}}sldIdLst")
    number = len(slide_ids) + 1
    source_slides = ET.fromstring(generated["ppt/presentation.xml"]).find(f"{{{P}}}sldIdLst")
    if len(source_slides) != 1:
        raise ValueError("The generated deck must contain exactly one slide.")

    mapping = {}
    for name in generated:
        if not name.endswith("/") and name.startswith(
            (
                "ppt/slides/",
                "ppt/slideLayouts/",
                "ppt/slideMasters/",
                "ppt/theme/",
                "ppt/media/",
                "ppt/notesSlides/",
                "ppt/notesMasters/",
            )
        ):
            folder, basename = posixpath.split(name)
            mapping[name] = posixpath.join(folder, f"acpdV3_{basename}")
    mapping["ppt/slides/slide1.xml"] = f"ppt/slides/slide{number}.xml"
    mapping["ppt/slides/_rels/slide1.xml.rels"] = f"ppt/slides/_rels/slide{number}.xml.rels"
    for name in list(mapping):
        if "/_rels/" in name:
            folder, basename = name.split("/_rels/")
            owning_part = f"{folder}/{basename[:-5]}"
            target_folder, target_basename = posixpath.split(mapping[owning_part])
            mapping[name] = f"{target_folder}/_rels/{target_basename}.rels"

    result = dict(original)
    for source, target in mapping.items():
        data = generated[source]
        if source.endswith(".rels"):
            relationships = ET.fromstring(data)
            owner_folder = posixpath.dirname(source.split("/_rels/")[0] + "/part")
            for relationship in relationships:
                if relationship.get("TargetMode") == "External":
                    continue
                linked = posixpath.normpath(posixpath.join(owner_folder, relationship.get("Target")))
                relationship.set(
                    "Target",
                    posixpath.relpath(mapping[linked], posixpath.dirname(target.split("/_rels/")[0] + "/part")),
                )
            data = xml_bytes(relationships)
        result[target] = data

    relationships = ET.fromstring(original["ppt/_rels/presentation.xml.rels"])
    next_id = max(int(item.get("Id")[3:]) for item in relationships if item.get("Id").startswith("rId")) + 1

    def add_relationship(kind: str, target: str) -> str:
        nonlocal next_id
        identifier = f"rId{next_id}"
        next_id += 1
        ET.SubElement(
            relationships, f"{{{REL}}}Relationship", {"Id": identifier, "Type": f"{R}/{kind}", "Target": target}
        )
        return identifier

    slide_relationship = add_relationship("slide", f"slides/slide{number}.xml")
    ET.SubElement(
        slide_ids,
        f"{{{P}}}sldId",
        {"id": str(max(int(item.get("id")) for item in slide_ids) + 1), f"{{{R}}}id": slide_relationship},
    )
    masters = presentation.find(f"{{{P}}}sldMasterIdLst")
    for source, target in mapping.items():
        if source.startswith("ppt/slideMasters/") and source.endswith(".xml"):
            relation = add_relationship("slideMaster", target.removeprefix("ppt/"))
            ET.SubElement(
                masters,
                f"{{{P}}}sldMasterId",
                {"id": str(max(int(item.get("id")) for item in masters) + 1), f"{{{R}}}id": relation},
            )
    result["ppt/presentation.xml"] = xml_bytes(presentation)
    result["ppt/_rels/presentation.xml.rels"] = xml_bytes(relationships)

    content_types = ET.fromstring(original["[Content_Types].xml"])
    defaults = {item.get("Extension") for item in content_types if item.tag == f"{{{CT}}}Default"}
    for item in ET.fromstring(generated["[Content_Types].xml"]):
        if item.tag == f"{{{CT}}}Default" and item.get("Extension") not in defaults:
            content_types.append(item)
            defaults.add(item.get("Extension"))
        if item.tag == f"{{{CT}}}Override" and item.get("PartName").lstrip("/") in mapping:
            item.set("PartName", "/" + mapping[item.get("PartName").lstrip("/")])
            content_types.append(item)
    result["[Content_Types].xml"] = xml_bytes(content_types)
    if "docProps/app.xml" in result:
        app = ET.fromstring(result["docProps/app.xml"])
        for item in app:
            if item.tag.endswith("}Slides"):
                item.text = str(number)
        result["docProps/app.xml"] = xml_bytes(app)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in result.items():
            archive.writestr(name, data)
    for name, data in original.items():
        if name.startswith(("ppt/slides/", "ppt/media/")):
            assert result[name] == data, name
    print(f"Appended slide {number}; original slide and media parts preserved: {args.output}")


if __name__ == "__main__":
    main()

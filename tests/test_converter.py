"""Tests for NEU-DET annotation conversion."""

from pathlib import Path

import pytest

from scripts.convert_annotations import parse_neu_det_xml, write_yolo_label
from visionguard.exceptions import AnnotationError
from visionguard.utils.dataset_utils import CLASS_TO_ID


def test_parse_neu_det_xml(tmp_path: Path) -> None:
    xml_content = """\
<annotation>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>
    <object>
        <name>inclusion</name>
        <bndbox>
            <xmin>50</xmin>
            <ymin>50</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    anns = parse_neu_det_xml(xml_path)
    assert len(anns) == 1
    ann = anns[0]
    assert ann["class_id"] == CLASS_TO_ID["inclusion"]
    assert ann["x_center"] == pytest.approx(0.5)
    assert ann["y_center"] == pytest.approx(0.5)
    assert ann["width"] == pytest.approx(0.5)
    assert ann["height"] == pytest.approx(0.5)


def test_parse_unknown_class(tmp_path: Path) -> None:
    xml_content = """\
<annotation>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>unknown</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_write_yolo_label(tmp_path: Path) -> None:
    annotations = [
        {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.25,
            "height": 0.25,
        }
    ]
    dst = tmp_path / "labels" / "test.txt"
    write_yolo_label(annotations, dst)

    content = dst.read_text(encoding="utf-8").strip()
    assert content.startswith("0 0.500000 0.500000 0.250000 0.250000")


def _write_xml(tmp_path: Path, content: str) -> Path:
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(content, encoding="utf-8")
    return xml_path


def test_parse_malformed_xml(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, "<annotation>")
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_missing_size(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
""",
    )
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_missing_width(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><height>100</height><depth>3</depth></size>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
""",
    )
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_missing_height(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><width>100</width><depth>3</depth></size>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
""",
    )
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_non_numeric_coordinate(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>foo</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
""",
    )
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_degenerate_box(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>60</xmin><ymin>0</ymin><xmax>50</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
""",
    )
    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_parse_class_name_uppercase_and_whitespace(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><width>200</width><height>200</height><depth>3</depth></size>
    <object>
        <name>  INCLUSION  </name>
        <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>150</xmax><ymax>150</ymax></bndbox>
    </object>
</annotation>
""",
    )
    anns = parse_neu_det_xml(xml_path)
    assert len(anns) == 1
    assert anns[0]["class_id"] == CLASS_TO_ID["inclusion"]


def test_parse_multiple_objects(tmp_path: Path) -> None:
    xml_path = _write_xml(
        tmp_path,
        """\
<annotation>
    <size><width>200</width><height>200</height><depth>3</depth></size>
    <object>
        <name>inclusion</name>
        <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>
    </object>
    <object>
        <name>scratches</name>
        <bndbox><xmin>120</xmin><ymin>120</ymin><xmax>180</xmax><ymax>180</ymax></bndbox>
    </object>
</annotation>
""",
    )
    anns = parse_neu_det_xml(xml_path)
    assert len(anns) == 2
    assert anns[0]["class_id"] == CLASS_TO_ID["inclusion"]
    assert anns[1]["class_id"] == CLASS_TO_ID["scratches"]

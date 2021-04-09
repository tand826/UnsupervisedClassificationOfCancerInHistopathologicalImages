from pathlib import Path
import xml.etree.ElementTree as ET

from colour import Color
import torch
from torch.utils.tensorboard import SummaryWriter


class Coords2ASAP:

    def __init__(self):
        self.ASAP_Annotations = ET.Element("ASAP_Annotations")
        self.Annotations = ET.SubElement(self.ASAP_Annotations, "Annotations")
        self.AnnotationGroups = ET.SubElement(
            self.ASAP_Annotations, "AnnotationGroups")

    def register_annotations(self, coords, groups, tile_size):
        """
        Args:
            coords (List[List[int, int]]): Coords of the patch.
            groups (List[int]): Class of the patch.
        """
        classes = set()
        idx = 0
        for (x, y), group in zip(coords, groups):
            idx += 1
            # save coords per polygons
            attribute_annotation = {
                "Name": f"Annotation {idx}",
                "Type": "Polygon",
                "PartOfGroup": str(group[0]),
                "Color": "#F4FA58"
            }
            Annotation = ET.SubElement(
                self.Annotations, "Annotation", attribute_annotation)
            coordinates = ET.SubElement(Annotation, "Coordinates")
            classes.add(str(int(group)))
            corners = [
                (x, y),
                (x + tile_size, y),
                (x + tile_size, y + tile_size),
                (x, y + tile_size)
            ]
            for order, corner in enumerate(corners):
                attribute_coord = {
                    "Order": str(order),
                    "X": str(corner[0]),
                    "Y": str(corner[1])
                }
                ET.SubElement(coordinates, "Coordinate", attribute_coord)

        colors = Color("blue").range_to(Color("green"), len(classes))
        classes = sorted(list(classes))
        for color, cls in zip(colors, classes):
            attributes_group = {
                "Name": cls,
                "PartOfGroup": "None",
                "Color": color.hex_l
            }
            Group = ET.SubElement(
                self.AnnotationGroups, "Group", attributes_group)
            ET.SubElement(Group, "Attributes")

    def save_xml(self, save_as):
        tree = ET.ElementTree(self.ASAP_Annotations)
        tree.write(save_as, xml_declaration=True)


class GatherASAP:

    def __init__(self):
        self.ASAP_Annotations = ET.Element("ASAP_Annotations")
        self.Annotations = ET.SubElement(self.ASAP_Annotations, "Annotations")
        self.annotations = list()
        self.AnnotationGroups = ET.SubElement(
            self.ASAP_Annotations, "AnnotationGroups")
        self.groups = set()

    def register_xml(self, xml):
        root = ET.parse(xml).getroot()
        for annot in root[0]:
            coords = []
            for i in range(4):
                coords.append({
                    "order": annot[0][i].attrib["Order"],
                    "x": annot[0][i].attrib["X"],
                    "y": annot[0][i].attrib["Y"]
                })
            self.annotations.append({
                "group": annot.attrib["PartOfGroup"],
                "coords": coords
            })

        for group in root[1]:
            self.groups.add(group.attrib["Name"])

    def gather(self):
        self._set_colors()
        self._gather_annotation()
        self._gather_group()

    def _set_colors(self):
        self.colors = Color("blue").range_to(Color("red"), len(self.groups))

    def _gather_annotation(self):
        annot_num = 1
        for annotation in self.annotations:
            attribute_annotation = {
                "Name": f"Annotation {annot_num}",
                "Type": "Polygon",
                "PartOfGroup": annotation["group"],
            }
            annot_num += 1
            Annotation = ET.SubElement(
                self.Annotations, "Annotation", attribute_annotation)
            coordinates = ET.SubElement(Annotation, "Coordinates")
            for coord in annotation["coords"]:
                attribute_coord = {
                    "Order": coord["order"],
                    "X": coord["x"],
                    "Y": coord["y"]
                }
                ET.SubElement(coordinates, "Coordinate", attribute_coord)

    def _gather_group(self):
        groups = sorted(map(int, self.groups))
        for color, group in zip(self.colors, groups):
            attributes_group = {
                "Name": str(group),
                "PartOfGroup": "None",
                "Color": color.hex_l
            }
            Group = ET.SubElement(
                self.AnnotationGroups, "Group", attributes_group)
            ET.SubElement(Group, "Attributes")

    def save_xml(self, save_as):
        tree = ET.ElementTree(self.ASAP_Annotations)
        tree.write(save_as, xml_declaration=True)


class SaveTo(type(Path())):

    def __init__(self, path):
        self.weight = self/"weight"
        self.chunk = self/"chunk"
        self.chunk.train = self.chunk/"train"
        self.chunk.test = self.chunk/"test"
        self.mkdir(exist_ok=True)
        self.weight.mkdir(exist_ok=True)
        self.chunk.mkdir(exist_ok=True)
        self.chunk.train.mkdir(exist_ok=True)
        self.chunk.test.mkdir(exist_ok=True)


class Checkpoint:

    def __init__(self, cwd, gpus, resume, save_to, save_model):
        self.save_to = save_to
        self.gpus = gpus
        self.resume = resume
        self.save_model = save_model

        if resume:
            self.save_to = SaveTo(cwd/resume)
        else:
            self.save_to = SaveTo(cwd/save_to)

        self.get_start()
        self.min_loss = {
            "value": float("inf"),
            "epoch": 0,
            "chunk": 0
        }

        self.writer = SummaryWriter("log")

    def save(self, model, optimizer, scheduler, epoch, chunk, loss=False):

        if not self.save_model:
            return

        if not loss:
            torch.save(
                scheduler.state_dict(),
                f"{self.save_to.weight}/scheduler{epoch:04}_{chunk}.pth")
            return

        if len(self.gpus) > 1:
            model = model.module
        torch.save(
            model.Encoder.state_dict(),
            f"{self.save_to.weight}/encoder{epoch:04}_{chunk}.pth")
        torch.save(
            model.Decoder.state_dict(),
            f"{self.save_to.weight}/decoder{epoch:04}_{chunk}.pth")

        torch.save(
            optimizer.state_dict(),
            f"{self.save_to.weight}/optimizer{epoch:04}_{chunk}.pth")

        torch.save(
            scheduler.state_dict(),
            f"{self.save_to.weight}/scheduler{epoch:04}_{chunk}.pth")

        with open(f"{self.save_to.weight}/last.txt", "w") as f:
            f.write(f"{epoch:04}_{chunk}")

        if loss < self.min_loss["value"]:
            self.min_loss = {
                "value": loss,
                "epoch": epoch,
                "chunk": chunk
            }
            torch.save(
                model.Encoder.state_dict(),
                f"{self.save_to.weight}/encoder_best.pt")
            torch.save(
                model.Decoder.state_dict(),
                f"{self.save_to.weight}/decoder_best.pt")
            with open(f"{self.save_to.weight}/best.txt", "w") as f:
                f.write(f"min loss = {loss} @ep{epoch:04} @chunk{chunk}")

    def load_state(self, model, optimizer, scheduler):
        with open(f"{self.save_to.weight}/last.txt", "r") as f:
            last = f.read().strip()

        if len(self.gpus) > 1:
            model.module.Encoder.load_state_dict(
                torch.load(f"{self.save_to.weight}/encoder{last}.pth"))
            model.module.Decoder.load_state_dict(
                torch.load(f"{self.save_to.weight}/decoder{last}.pth"))
        else:
            model.Encoder.load_state_dict(
                torch.load(f"{self.save_to.weight}/encoder{last}.pth"))
            model.Decoder.load_state_dict(
                torch.load(f"{self.save_to.weight}/decoder{last}.pth"))

        optimizer.load_state_dict(
            torch.load(f"{self.save_to.weight}/optimizer{last}.pth"))

        scheduler.load_state_dict(
            torch.load(f"{self.save_to.weight}/scheduler{last}.pth"))

        return model, optimizer, scheduler

    def get_start(self):
        if self.resume:
            with open(f"{self.save_to.weight}/last.txt", "r") as f:
                last = f.read().strip()
            last_ep, last_chunk = list(map(int, last.split("_")))
            chunks = len(list(self.save_to.chunk.glob("chunk*.csv")))
            if last_chunk == chunks:
                self.start_chunk = 0
                self.start_epoch = last_ep + 1
            else:
                self.start_chunk = last_chunk + 1
                self.start_epoch = last_ep
        else:
            self.start_chunk = 0
            self.start_epoch = 1

    def log(self, tag, loss, epoch, chunk, chunks, batch_idx, niter_batch):
        niter_chunk = chunk * niter_batch
        niter_epoch = (epoch - 1) * chunks * niter_batch
        niter = niter_epoch + niter_chunk + batch_idx
        if niter > 1000:
            import sys
            sys.exit()
        self.writer.add_scalar(tag, loss.item(), niter)

    def close_writer(self):
        self.writer.close()

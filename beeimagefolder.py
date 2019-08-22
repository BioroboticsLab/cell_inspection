import torchvision
import os, sys
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, make_dataset

# BeeImageFolder = DatasetFolder + ImageFolder
# accepts images like ImageFolder
# valid_classes can be defined manually

class BeeImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, valid_classes=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root, valid_classes=valid_classes)
        extensions = torchvision.datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def _find_classes(self, dir, valid_classes):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (valid_classes is None or d.name in valid_classes)]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
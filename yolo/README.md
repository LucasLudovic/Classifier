# Classification YOLO26

Meme pipeline que le CNN maison, mais avec un backbone YOLO26 pre-entraine sur
ImageNet. Ultralytics gere la boucle d'entrainement, l'augmentation et le
checkpointing : le `Trainer` n'est qu'une facade au-dessus.

## Installation

```bash
pip install -r yolo/requirements.txt
```

## Utilisation

A lancer depuis la racine du repo, en `-m` : les imports du package sont
absolus (`yolo.*`), donc `python yolo/train.py` ne resoudrait pas.

```bash
python -m yolo.train      # entrainement + eval sur test
python -m yolo.predict    # inference
```

Les reglages sont dans `yolo/config.py` : `HyperParameters` pour
l'entrainement, `InferenceParameters` (poids, source) pour l'inference.

## Format du dataset

Un dossier par split, un sous-dossier par classe. Pas de fichier de labels :
le nom du dossier *est* le label.

```
data/
├── train/
│   ├── none/       *.jpg
│   └── electrod/   *.jpg
├── val/
│   ├── none/
│   └── electrod/
└── test/           (optionnel)
    ├── none/
    └── electrod/
```

Sur Roboflow : projet de type **Single-Label Classification**, export en
**Folder Structure**. Roboflow nomme le split de validation `valid/`, alors
qu'Ultralytics attend `val/` — `yolo/dataset.py` fait le renommage
automatiquement au demarrage de `main.py`.

Split conseille : 70 / 20 / 10 (train / val / test), stratifie par classe, et
les images d'une meme serie (memes conditions de prise de vue) doivent rester
dans le meme split pour eviter les fuites.

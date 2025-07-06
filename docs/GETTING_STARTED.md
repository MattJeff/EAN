# HyperCubeX – Getting Started (Rotation Curriculum)

Bienvenue ! Cette courte documentation vous guide pour lancer les expériences de rotation 3×3 avec l’optimiseur REINFORCE.

## Installation rapide

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # minimal numpy, pytest…
```

## Exécution d’un curriculum 3 × 3

```bash
python experiments/curriculum_rotation3x3.py data/rotation3x3_dummy.json 90 100
```

Un fichier `rotation3x3_log.csv` est créé avec :

| tick | n_assemblies | mean_energy_global | reward |
|------|--------------|--------------------|--------|

## Changer d’optimiseur

`PolicyScheduler` accepte :

* `optimizer="reinforce"` – new policy-gradient (par défaut).
* `optimizer="perturb"` – recherche gloutonne aléatoire.

## Diagramme d’architecture (ASCII)

```text
                +------------------+
                |  Experiments/    |  <--- CLI scripts
                +---------+--------+
                          |
                          v
+-----------+    +------------------+    +-------------------+
|  Teachers |--->|  PolicyScheduler |<---|  Optimizers        |
| Rotation  |    |  (hx_core)       |    |  Reinforce / Pert |
+-----------+    +---------+--------+    +----+--------------+
                          |                   |
                          v                   |
                    +-----+------+            |
                    | Adapter    |            |
                    | (hx_adapter)|            |
                    +-----+------+            |
                          |                   |
                          v                   |
                     +----+--------------------+
                     |  Core Network (SNN)    |
                     +-------------------------+
```

---

Pour plus de détails, consultez le code source et les tests unitaires (`pytest`).

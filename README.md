# AMR-Scope AI — prototype hackathon

Application web (mobile-first, en français) qui analyse une photo d'antibiogramme (boîte de Pétri)
avec **OpenCV** : détection de la boîte, des disques d'antibiotique, mesure des diamètres
d'inhibition (mm), interprétation S / I / R, et validation humaine obligatoire en cas de doute.

## Lancer

```bash
pip install -r requirements.txt
python app.py
# ouvrir http://localhost:5000  (bouton « Image d'exemple » pour tester sans photo)
```

Sur téléphone (même Wi-Fi) : ouvrir `http://<IP-de-l'ordinateur>:5000`.

## Structure

| Fichier | Rôle |
|---|---|
| `analyzer.py` | Pipeline OpenCV (HoughCircles, profil radial, score de confiance) |
| `app.py` | API Flask : `POST /api/analyze`, `GET /api/breakpoints` |
| `static/index.html` | Interface en 4 étapes (acquisition, analyse, vérification, enregistrement) |
| `data/` | Créé automatiquement : `amr.db` (SQLite, résultats) + `images/` (photos originales) |
| `make_sample.py` | Génère une image synthétique de test |

## Stockage

Chaque analyse validée est enregistrée dans `data/amr.db` (table `analyses` : date, site, résultats S/I/R, drapeau validation manuelle) et la photo originale dans `data/images/<id>.jpg`. Le bouton « Historique » de l'étape 1 les relit. Pour tout effacer : supprimer le dossier `data/`.

## Comment ça mesure

1. La boîte (90 mm) est détectée par transformée de Hough → échelle px/mm.
2. Les disques (6 mm) sont détectés, puis filtrés (plus clairs que leur voisinage).
3. Pour chaque disque : profils radiaux sur 45 angles, position du plus fort gradient = bord de la zone,
   médiane des angles = rayon. Diamètre = 2 × rayon (disque inclus).
4. Confiance = dispersion entre angles + contraste du bord + couverture. Sous 75 % → « Mesure incertaine »,
   case « Confirmer la mesure manuelle » obligatoire.

## Limites connues (à dire au jury)

- Testé sur images **synthétiques** uniquement ; à valider sur de vraies boîtes (reflets, ombres, pelouse irrégulière).
- Seuils S/I/R de `app.py` = **valeurs de démonstration**, pas les vrais seuils EUCAST (qui dépendent de l'espèce et de la version du tableau).
- Les étiquettes d'antibiotiques sont attribuées par défaut puis corrigées à l'étape 3 (pas d'OCR des disques).
- Outil d'aide à la décision : un expert doit valider.

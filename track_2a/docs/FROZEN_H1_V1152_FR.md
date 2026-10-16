# CLAUSESHIFT V1.15.2 — Correctif de complétude pour documents longs

## Motivation (incident de livraison, non optimisation des scores)

L'exécution V1.15.1 s'est arrêtée correctement sur une entrée dépassant le plafond **par inférence** de 3 072 sous-tokens. Elle n'a pas exporté de document incomplet. **Aucune annotation finale et aucun score final n'ont été consultés pour créer ce correctif.** Il faut signaler explicitement dans le rapport que ce correctif a été développé *après* la première exécution finale, pour un motif de complétude technique. Vérifier que cette procédure reste autorisée par les règles de la compétition.

V1.15.2 conserve la limite de **3 072 sous-tokens par appel mmBERT**, sans padding, troncature ni prédiction inventée. Les documents qui tiennent dans la limite utilisent **strictement l'ancien chemin d'inférence (un seul `model.predict`)**. Un document dépassant la limite est découpé de façon déterministe par *positions relatives des mots* des deux langues en N passages parallèles. Chaque passage contient 24 mots de contexte supplémentaire à gauche/droite dans chaque langue (si disponibles). Chaque token central est prédit **une seule fois** par le modèle ; les scores du contexte chevauchant ne sont pas exportés. Les sorties sont réassemblées exactement dans l'ordre des tokens originaux. Un maximum de 64 passages protège contre les déséquilibres extrêmes. Toute lacune, dérive d'ordre, non-finitude ou dépassement du budget **bloque l'export**.

Limite méthodologique : le découpage aux positions relatives est une approximation des correspondances bilingues ; il peut modifier les scores aux frontières et aucun gain scientifique n'est revendiqué. Aucun paramètre H1 (18 documents/langue, 54 appels au plus, fenêtre 32, 12 indices/côté, fusion 0,35) ni modèle n'est modifié. Les performances publiées sont celles de V1.15.1 DEV, pas un score V1.15.2 réévalué sur tout DEV.

## Installer sur le VPS (avant tout nouvel accès final)

1. **Archiver le gel précédent** et les traces. Ne pas supprimer `out/`, `.hf_cache/` ou les checkpoints existants, ni lire les annotations finales :

```bash
cd /var/www/clauseshift_track2a
mkdir -p /root/clauseshift-evidence/incident-v1151
cp -p docs/submission_frozen.json /root/clauseshift-evidence/incident-v1151/submission_frozen.json
cp -p /root/clauseshift-h1-final.log /root/clauseshift-evidence/incident-v1151/ 2>/dev/null || true
git rev-parse HEAD > /root/clauseshift-evidence/incident-v1151/git_commit.txt
```

2. Transférer `CLAUSESHIFT_H1_Long_Document_Completion_v1.15.2.patch` et appliquer après contrôle :

```bash
git apply --check CLAUSESHIFT_H1_Long_Document_Completion_v1.15.2.patch
git apply CLAUSESHIFT_H1_Long_Document_Completion_v1.15.2.patch
python -m unittest discover -s tests -q
```

Le résultat attendu est **116 tests** (112 précédents + 4 nouveaux). Le manifeste **V1.15.1** n'est volontairement plus valide après changement du code. Ne pas contourner le préflight.

3. Tester le découpage **sans aucun document final**. Le test unitaire vérifie réassemblage exact et absence de checkpoint partiel sur des exemples synthétiques. Il est également recommandé de vérifier une inférence mmBERT réelle sur un **texte synthétique** dépassant volontairement le cap, avec les caches existants, dans Docker avant le gel.

4. Réaliser un **nouveau gel documenté** (le précédent reste archivé). Important : effectuer le commit du code AVANT la commande de gel, et reconstruire Docker APRÈS son écriture. Ne jamais éditer le manifeste manuellement :

```bash
# Archiver l'ancien gel hors du dépôt ; son retrait fait partie du commit de correctif.
rm docs/submission_frozen.json
git add src/h1_hybrid.py src/submission.py tests/test_h1_frozen_v115.py \
  tests/test_h1_chunked_v1152.py docs/FROZEN_H1_V1152_FR.md \
  README.md technical_report.md
# Inclure explicitement la suppression de l'ancien manifeste.
git add -u docs/submission_frozen.json
git diff --cached --check
git diff --cached --stat
git commit -m 'H1 completeness hotfix: deterministic over-cap windows'

export EMB_SHA='c5955035435e2bf121cde7f3c8863ef52ff35d82'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
make freeze METHOD=H1-HYBRID \
  UPSTREAM_SHA=1807a42100e742ed03d337c54c4b9ea86995f565 \
  FROZEN_COMMIT="$(git rev-parse HEAD)" EMB_SHA="$EMB_SHA"
git add docs/submission_frozen.json
git commit -m 'Freeze H1 V1.15.2 after completeness hotfix'
make preflight
make build
```

5. Avant `make run`, réaliser la vérification des secrets via le conteneur et le contrôle offline `preflight`. Le cache mmBERT épinglé est réutilisé par le montage `.hf_cache` ; un nouveau hash de code entraîne de **nouveaux checkpoints V1.15.2** : ne pas falsifier les signatures pour reprendre les checkpoints V1.15.1. Ne pas exporter tant que les trois langues et H1 ne sont pas complets.

## À signaler dans le rapport

- Premier gel V1.15.1, refus d'un document dépassant le cap ; aucun score final évalué ou utilisé pour optimiser la méthode.
- Correctif V1.15.2 : politique fixe de découpage des textes longs en fenêtres bilingues, contexte +/-24 mots, plafond de 3 072 sous-tokens **par passage**, reconstruction exacte avec refus sur écart.
- Les documents courts restent sur le chemin original ; pas de nouvelle calibration du scoring ni d'accès au gold final.
- Le premier gel et le nouveau gel sont horodatés / identifiés par les commits.
- Vérifier les règles officielles : une modification de méthode après avoir accédé au jeu final peut nécessiter une déclaration et peut ne pas être permise, même si elle ne touche pas aux scores.

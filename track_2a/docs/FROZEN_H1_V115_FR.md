# CLAUSESHIFT V1.15 — Frozen H1 Hybrid Runner (Track 2A)

## État réel au 15 octobre 2026

- Baseline validée sur DEV officiel, 168 documents par langue : DE 0,146786 ; FR 0,087821 ; IT 0,152947 ; macro 0,129185.
- Ablation H1 sélection fixée, 18 documents/langue, **pas** ensemble DEV complet : baseline 0,106802 ; hybride 0,107801 ; delta +0,001000. 54 tentatives (44 réponses acceptées, 10 replis), 1 056 scores modifiés ; FR régresse.
- Le code V1.15 réimplémente la même fenêtre, sélection et le **même prompt H1** en calculant la baseline depuis les textes **uniquement**.
- Tests : 112 tests locaux au moment de la livraison, incluant 5 tests H1 synthétiques. **Ni build Docker sur le VPS, ni vraie exécution V1.15 depuis un checkout propre, ni exécution finale ne sont encore validés.**
- Les références DiffAlign `parallel_all_1epoch` (0,2213 macro) portent sur un checkpoint distinct qui n'est pas dans ce livrable.

## Comportement figé

- Upstream SwissGov-RSD `1807a42100e742ed03d337c54c4b9ea86995f565` ; Docker clone le dépôt en *sparse checkout* `rsd` seulement, sans données gold dans l'image.
- mmBERT public `jhu-clsp/mmBERT-base`, snapshot `c5955035435e2bf121cde7f3c8863ef52ff35d82`, DiffAlign couche `-1`, CPU.
- Limite prudente `max_subtokens=3072` par côté. DEV connu passe ; le final n'a pas été inspecté. Tout dépassement **bloque** l'export et nécessite un nouveau cycle de décision, non un padding arbitraire.
- H1 : 18 documents sélectionnés par langue sur longueur; fenêtres 32 tokens; jusqu'à 12 indices par côté, **un appel/document, au plus 54**. Sélection déterministe depuis textes et prédictions mmBERT, sans gold.
- Apertus `swiss-ai/apertus-v1.5-8b`, `https://api.publicai.co/v1`, température 0, `max_tokens=600`, timeout 52 s, 0 retry automatique. Fusion : `0,65*baseline + 0,35*(1 - similarity/5)` pour les seuls indices validés.
- Chaque tentative est pré-réservée atomiquement ; après interruption elle ne sera pas facturée de nouveau automatiquement. Les réponses non conformes sont journalisées en `error`, et le score mmBERT est conservé sur les tokens correspondants. Au moins une modification réussie est exigée pour exporter comme méthode H1.
- Un document de prédiction n'est jamais exporté s'il manque un autre document/langue ou l'un des 54 statuts H1. L'export suit l'ordre de l'input par langue.
- L'accès aux entrées *full/final* est exclusivement effectué via `src.submission run` **après** vérification du manifeste gelé. Aucun *gold* n'est utilisé dans le moteur H1.

## Installation sur le nouveau Droplet (sans toucher à devtest2)

1. Télécharger et transférer `CLAUSESHIFT_Track2A_Implementation_v1.15_FrozenH1.zip` vers `/root/` du VPS.
2. Extraire dans `/var/www` :

```bash
cd /var/www
unzip /root/CLAUSESHIFT_Track2A_Implementation_v1.15_FrozenH1.zip
mv CLAUSESHIFT_Track2A clauseshift_track2a
cd clauseshift_track2a
source /opt/clauseshift-venv/bin/activate
python -m unittest discover -s tests -q
```

3. Tester un build sans consulter le final :

```bash
make build
make warmup    # Télécharge seulement le modèle mmBERT révision épinglée dans .hf_cache/
python -m src.submission preflight  # DOIT échouer tant que le manifeste n'existe pas
```

La construction Docker exige l'accès sortant à PyPI, Hugging Face, Docker Hub et GitHub. **Sur cette livraison, le build Docker sur le VPS n'a pas encore été confirmé.**

## Gel — uniquement après inspection DEV, tests et validation de faisabilité

Ne pas geler en présence d'une anomalie bloquante. Le dépôt doit contenir le code, la licence et les dépendances ; aucun export DEV, gold, cache, secret ou fichier final.

```bash
cd /var/www/clauseshift_track2a
# Réutiliser ton dépôt Git si présent, sinon initialiser un nouveau dépôt.
[ -d .git ] || git init
git add .
git commit -m 'CLAUSESHIFT frozen H1 implementation'
export EMB_SHA='c5955035435e2bf121cde7f3c8863ef52ff35d82'
export LLM_NAME='swiss-ai/apertus-v1.5-8b'
export LLM_BASE_URL='https://api.publicai.co/v1'
make freeze METHOD=H1-HYBRID \
  UPSTREAM_SHA=1807a42100e742ed03d337c54c4b9ea86995f565 \
  FROZEN_COMMIT="$(git rev-parse HEAD)" EMB_SHA="$EMB_SHA"
git add docs/submission_frozen.json
git commit -m 'Freeze H1 model, source, and runtime protocol'
make preflight
```

Ne pas modifier le code, le Dockerfile, les dépendances, le Makefile ni les scores/paramètres après gel : preflight refuse toute altération. Contrôler la clé dans ce **même terminal** :

```bash
IFS= read -r -s -p 'Clé Public AI : ' LLM_API_KEY; echo
export LLM_API_KEY
python -m src.submission preflight
```

Le second preflight sans `--require-secrets` ne confirme pas l'authentification ; le run vérifie la clé et l'URL et refuse toute valeur contenant des caractères de contrôle.

## Run final, exclusivement après gel

```bash
cd /var/www/clauseshift_track2a
mkdir -p .hf_cache
# Préserve les checkpoints `out/` et les poids `.hf_cache/` entre deux conteneurs.
set -o pipefail
make run 2>&1 | tee /root/clauseshift-h1-final.log
```

La commande produit sous `out/predictions/` les trois fichiers attendus `CLAUSESHIFT_admin_{de,fr,it}.jsonl.jsonl`, uniquement après validation de la complétude. `make run` ne doit pas être lancé comme test préalable : il donne accès à la partition finale après gel.

À contrôler avant dépôt :

```bash
python -m src.cli validate --split final
sha256sum out/predictions/CLAUSESHIFT_admin_*.jsonl.jsonl
```

**Si le final dépasse 3 072 sous-tokens ou qu'un état H1 demeure absent, aucune solution n'est implicitement fabriquée** ; le run s'arrête. Ne pas corriger les paramètres en se basant sur les données finales. En cas d'échec dû à la connectivité, le run peut être repris avec les mêmes checkpoints et le même manifeste. Les statuts `started`/`error` restent des replis non réémis, conformément au protocole sans retries.

## Attention : exigences de soumission

Ne pas annoncer le delta H1 comme un gain global du jeu DEV. Les résultats mmBERT complets et les résultats H1 à 54 documents utilisent des périmètres différents. La soumission doit détailler clairement l'utilisation d'Apertus, les 10 échecs H1 DEV, les limites et l'absence de garantie de supériorité en dehors de DEV.

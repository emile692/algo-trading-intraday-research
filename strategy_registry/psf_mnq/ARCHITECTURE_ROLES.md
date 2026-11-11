# Tri par rôle architectural

Cette passe classe les 134 stratégies de l'univers décidé ensemble selon leur
meilleur rôle dans une architecture de recherche modulaire. Elle ne supprime
aucune stratégie et ne constitue pas une validation de performance.

## Règle de classement

Chaque stratégie reçoit un rôle principal unique :

| Rôle | Question opérationnelle |
|---|---|
| `alpha_engine` | Quelle hypothèse directionnelle doit produire le PnL ? |
| `setup_trigger` | Quel événement local déclenche ou confirme l'entrée ? |
| `filter_regime` | Dans quel état de marché le moteur peut-il être activé ? |
| `level_context` | Autour de quel niveau ou de quelle zone intervient-on ? |
| `risk_overlay` | Comment adapte-t-on le stop ou la sortie après l'entrée ? |

Le rôle décrit l'usage de recherche retenu, même lorsque la fiche source
présente l'indicateur comme une stratégie autonome.

## Première passe à revoir ensemble

| Rôle | Nombre |
|---|---:|
| `alpha_engine` | 43 |
| `setup_trigger` | 62 |
| `filter_regime` | 17 |
| `level_context` | 10 |
| `risk_overlay` | 2 |
| **Total** | **134** |

Le détail ligne par ligne est dans `architecture_roles.csv`. Toutes les lignes
portent le statut `draft_to_review` afin que les arbitrages restent explicites.

La [revue logique des 43 moteurs provisoires](ALPHA_ENGINE_LOGICAL_AUDIT.md)
réduit ensuite ce compartiment à **35 moteurs autonomes uniques**, regroupés
en **16 mécanismes**. Elle conserve séparément 19 variantes, déplace trois
lignes vers les triggers et deux vers les filtres, fusionne deux doublons et
rejette une fiche insuffisamment définie.

Ces 16 mécanismes sont maintenant traduits en
[spécifications canoniques figées](FROZEN_BASELINES_REVIEW.md), accompagnées
du benchmark indépendant `MNQ_BUY_AND_HOLD`. Cette étape ne lance aucun
backtest.

## Lecture du résultat

- Les 62 `setup_trigger` ne sont pas traités comme 62 moteurs d'alpha
  indépendants. Ils pourront être testés sur un nombre plus réduit de moteurs.
- Les deux `risk_overlay` sont les variantes Parabolic SAR, dont la fonction
  réutilisable la plus distinctive est le stop suiveur dynamique.
- Les méthodes de money management déjà retirées ne sont pas réintroduites.
- Les rôles ne préjugent ni de la qualité statistique ni de la priorité de test.

## Reproduction

```powershell
uv run python strategy_registry/psf_mnq/classify_architecture_roles.py
```

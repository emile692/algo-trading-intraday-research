# Run IS n°1 — invalidé (bug du verrou de gain)

Run du 2026-11-12, conservé pour audit. Ne pas utiliser.

**Bug.** `SessionTable.from_results` complétait les tableaux de records d'équité avec `+inf`. Pour une séance qui n'atteignait jamais le seuil du verrou de gain, la recherche du premier record ≥ seuil tombait donc sur une case de remplissage, valant 0 USD. Résultat : la configuration `B1k0.40+lock` affichait 0 % de PASS, ce qui est absurde.

**Portée.** Seules les configurations avec verrou de gain sont touchées. Les autres n'appellent jamais la recherche dans les records (seuil infini). Leurs chiffres sont identiques dans le run n°2.

**Correction.** Remplissage à `-inf`, plus deux tests : `test_T4_profit_lock_padding_never_triggers` et `test_lifecycle_profit_lock_still_passes`. Self-test et sélection IS relancés avec le code corrigé, avant toute lecture de l'OOS.

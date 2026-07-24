PAIRED WILCOXON ANALYSIS OUTPUTS
================================

1. all_episode_results_long.csv
   Raw episode-level data from every architecture and every independent run.

2. seed_averaged_episode_results.csv
   Three-run mean and standard deviation for each architecture, topology,
   episode_id, and start_index.

3. paired_costs_wide.csv
   One row per test day, with architecture costs in separate columns.

4. paired_daily_cost_differences.csv
   Direct GNN-vs-MLP matched pairs. The column
   difference_gnn_minus_mlp is negative when the GNN is cheaper.

5. wilcoxon_cost_results.csv
   Per-feeder, per-topology statistical results:
   - two-sided paired Wilcoxon p-value
   - one-sided p-value for GNN cost < MLP cost
   - Holm-adjusted p-values across GCN/GAT/TAGConv
   - GNN win counts and win rates
   - mean and median paired differences
   - percentage cost improvement
   - matched-pairs rank-biserial effect size

6. design_summary.csv
   Checks expected runs and episode counts.

7. pairing_alignment_check.csv
   Confirms that MLP and each GNN share identical
   (episode_id, start_index) pairing keys.

8. loaded_file_report.csv
   Inventory of every loaded episode_summary.csv.

Recommended paper reporting
---------------------------
Use the two-sided Holm-adjusted p-value as the conservative primary result.
The directional one-sided result may be included only if the hypothesis
"GNN cost is lower than MLP cost" was specified before examining the data.

Interpretation
--------------
difference_gnn_minus_mlp < 0  : GNN is cheaper
rank_biserial < 0             : effect favors GNN

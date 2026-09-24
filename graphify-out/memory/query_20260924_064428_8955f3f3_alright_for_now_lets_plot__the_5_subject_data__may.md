---
type: "query"
date: "2026-09-24T06:44:28.608422+00:00"
question: "alright for now lets plot, the 5 subject data, maybe a bar plot, for each subject, and their corresponding, error, say sub1, sub2, no names."
contributor: "graphify"
outcome: "useful"
source_nodes: ["agreement()", "Plot"]
---

# Q: alright for now lets plot, the 5 subject data, maybe a bar plot, for each subject, and their corresponding, error, say sub1, sub2, no names.

## Answer

Expanded via graph vocabulary: [agreement, axial, axis, bars, csv, error, errors, lateral, plot, rmse]. Added an anonymized grouped RMSE bar chart to 16_subject_agreement.py. It labels the configured five subjects Sub1 through Sub5 and shows flexion, lateral, and axial RMSE in degrees with numeric labels, while printing the private mapping only to the console. Generated data/sep23_26/output/subject_rmse_bar.png and verified its rendering; Ruff and git diff checks pass.

## Outcome

- Signal: useful

## Source Nodes

- agreement()
- Plot
# Optional confirmatory sensitivity analysis. Review with a biostatistician.
# Install once in R: install.packages(c("lme4", "broom.mixed", "readr", "dplyr"))

library(lme4)
library(broom.mixed)
library(readr)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scripts/11_sensitivity_models.R <final_dataset.csv> <output_dir>")
}

data_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dat <- read_csv(data_path, show_col_types = FALSE) %>%
  mutate(
    top1_correct = as.integer(top1_correct),
    unsafe_management = as.integer(unsafe_management),
    anchor_condition = factor(anchor_condition, levels = c("no_label", "correct_label", "incorrect_label")),
    logical_model_id = factor(logical_model_id),
    domain = factor(domain),
    case_id = factor(case_id)
  )

part1 <- dat %>% filter(part == "part1")

accuracy_model <- glmer(
  top1_correct ~ anchor_condition * logical_model_id + domain + (1 | case_id),
  data = part1,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

unsafe_model <- glmer(
  unsafe_management ~ anchor_condition * logical_model_id + domain + (1 | case_id),
  data = part1,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

write_csv(tidy(accuracy_model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE),
          file.path(output_dir, "mixed_logistic_top1_accuracy.csv"))
write_csv(tidy(unsafe_model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE),
          file.path(output_dir, "mixed_logistic_unsafe_management.csv"))

capture.output(summary(accuracy_model), file = file.path(output_dir, "mixed_logistic_top1_accuracy.txt"))
capture.output(summary(unsafe_model), file = file.path(output_dir, "mixed_logistic_unsafe_management.txt"))


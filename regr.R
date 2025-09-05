library(dplyr)
library(brms)
options(mc.cores = parallel::detectCores())
options(browser = "false")

check_all_diagnostics <- function(m) {
rstan::check_hmc_diagnostics(m$fit)

print(ifelse(sum(rhat(m) > 1.05) > 1, 'Some R-hat > 1.05', 'No R-hat > 1.05'))

print(ifelse(sum(m$criteria$loo$diagnostics$pareto_k > 0.7), 'Some pareto-k > 0.7', 'No pareto-k > 0.7'))
}

models_path <- 'NormaUsInterferencia/resultats/'

monolings <- c('BSC-LT/RoBERTa-ca', 'catallama/CataLlama-v0.2-Base', 'gplsi/Aitana-6.3B', 'catallama/CataLlama-v0.1-Base')

estr_interferencia <- c('caiguda_prep', 'a_od', 'com_reg', 'alt_prep')

df <- readr::read_csv('NormaUsInterferencia/resultats/resultats.csv') %>%
      mutate(deltaNLL      = pno_normatiu - pnormatiu, 
             interferencia = ifelse(tipus %in% estr_interferencia, 'sí', 'no'), 
             multiling      = ifelse(model %in% monolings, 'no', 'sí'),
             normHLL = ifelse(pnormatiu < pno_normatiu, 1, 0)) %>% 
             filter(model != 'BSC-LT/RoBERTa-ca') %>% #excluding masked language models since the task is different and delta (pseudo) NLL reflects the higher uncertainty in repeated cloze
             filter( model != 'catallama/CataLlama-v0.2-Base') #excluding CataLlama-v0.2 since it is a merge of CataLlama-v0.1 with Llama Instruct

m <- brm(data = df, 
          formula = normHLL ~ 1 + multiling * interferencia + ( 1 | index) + (1 | model),
          family = bernoulli(),
          file     = paste0(models_path, 'm'),
          iter = 2000,
          seed =25,
          control = list(adapt_delta=0.99),
          save_pars = save_pars(all = TRUE)
          )
m <- add_criterion(m, 'loo', moment_match = TRUE)
check_all_diagnostics(m)

df_pp <- data.frame('multiling' = c('sí', 'sí', 'no', 'no'),
                             'interferencia' = c('sí', 'no', 'sí', 'no')
                             )
pp <- predict(m, df_pp, re_formula = NA) %>% data.frame()# %>% rbind(df_pp)
cbind(df_pp, pp)

library(ggplot2)
library(ggokabeito)
library(patchwork)

p_multiling <- plot(conditional_effects(m, effects=c('multiling')))[[1]]
p_interf <- plot(conditional_effects(m, effects=c('interferencia')))[[1]]
p_interaction <- plot(conditional_effects(m, effects=c('multiling:interferencia')))[[1]]


 p_interaction <- p_interaction + 
 ylab('Probabilitat( normatiu )') +
 theme_minimal(base_size = 25) +
   scale_x_discrete(labels = c(
    "sí" = "Multilingüe",
    "no" = "Monolingüe"
   )) + 
 theme(legend.position = c(0.5,0.85),
      legend.box.background=element_rect(fill="white"),
      legend.text=element_text(size=13),
      legend.title=element_text(size=13),
      axis.title.x=element_blank()
      )  +
 scale_color_okabe_ito() +
 labs(color = 'Interferència', fill = 'Interferència')

p_multiling <- p_multiling + 
ylab('Probabilitat( normatiu )') +
 theme_minimal(base_size = 25) +
   scale_x_discrete(labels = c(
    "sí" = "Multilingüe",
    "no" = "Monolingüe"
   )) + 
 theme(legend.position = c(0.85,0.85),
      legend.box.background=element_rect(fill="white"),
      legend.text=element_text(size=18),
      legend.title=element_text(size=18),
      axis.title.x=element_blank()
      ) + 
      ylim(0,1)

p_interf <- p_interf + ylab('Probabilitat( normatiu )') +
 theme_minimal(base_size = 25) +
   scale_x_discrete(labels = c(
    "sí" = "Interferència",
    "no" = "Sense interferència"
   )) + 
 theme(legend.position = c(0.85,0.85),
      legend.box.background=element_rect(fill="white"),
      legend.text=element_text(size=18),
      legend.title=element_text(size=18),
      axis.title.x=element_blank()
      ) + 
      ylim(0,1)


library(tibble)
library(gridExtra)
library(grid) # For gpar()

feff <- as.data.frame(fixef(m)) %>%
  rownames_to_column("Paràmetre") %>%
  rename(Estimat = 'Estimate', Error.Est = "Est.Error") %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

feff$Paràmetre <- c('Intersecció', 'Multilingüe', 'Interferència', 'Interferència i Multilingüe')


my_theme <- ttheme_minimal(
  core = list(
   fg_params = list(hjust = 0, x = 0.05) # Apply to all body cells
  ),
  base_size = 15
  )

p_table <- tableGrob(feff, rows = NULL, theme = my_theme)

top_row <- p_multiling + p_interf
bottom_row <- p_interaction + p_table 
pwork <- top_row / bottom_row

pwork + plot_annotation(tag_levels = 'A') & 
  	  theme(plot.tag = element_text(size = 35))
ggsave('NormaUsInterferencia/resultats/fig-model.pdf', dpi = 1200, width = 35, height = 30, units = "cm", device = cairo_pdf)




### Including excluded models

df <- readr::read_csv('NormaUsInterferencia/resultats/resultats.csv') %>%
      mutate(deltaNLL      = pno_normatiu - pnormatiu, 
             interferencia = ifelse(tipus %in% estr_interferencia, 'sí', 'no'), 
             multiling      = ifelse(model %in% monolings, 'no', 'sí'),
             normHLL = ifelse(pnormatiu < pno_normatiu, 1, 0)) 


m.ex <- brm(data = df, 
          formula = normHLL ~ 1 + multiling *interferencia + ( 1 | index) + (1 | model),
          family = bernoulli(),
          file     = paste0(models_path, 'mex'),
          iter = 2000,
          seed =24,
          control = list(adapt_delta=0.99),
          save_pars = save_pars(all = TRUE)
          )
m.ex <- add_criterion(m.ex, 'loo', moment_match = TRUE)
check_all_diagnostics(m.ex)

library(kableExtra)

fixef(m.ex) %>%
  round(2) %>%
  as.data.frame() %>%
      kbl(row.names = FALSE, format = 'latex', booktabs = TRUE, linesep = '') %>%
      kable_styling(latex_options = 'striped') 

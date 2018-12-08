#Saptarshi Ghose + Natasha Mathur
#Science of Elections and Campaigns
#Final Project

rm(list=ls())   

#Set working directory
setwd("~/Downloads/")

#Install libraries
install.packages('plyr')
install.packages('lfe')
install.packages('stargazer')

#Load libraries
library(tidyverse)
library(ggplot2)
library(haven)
library(plyr)
library(stargazer)

options(scipen=999)

#Import dta file
data <- read.csv(file = "final_data_for_model_1.csv")

#Attach the data
attach(data)

#models
final_model <- lm(vote_share ~ pro_gun_control + candidate_male + candidate_hispanic + prog_min_wage + incumbent + factor(year) + factor(office) + factor(county) + factor(district))

no_iv_model <- lm(vote_share ~ pro_gun_control + candidate_male + candidate_hispanic + factor(year) + factor(office) + factor(county) + factor(district))

stargazer(no_iv_model, final_model, type = 'text', style = 'jpam')



###################################################
###################################################
## ABCD Neurocognitive Prediction Challenge      ##
## Computing Validation MSE & R-squared          ##
##                                               ##
## Feb 13, 2019                                  ##
##                                               ##
###################################################
###################################################

## Read in and 'ground-truth' scores and predicted scores from contestant, should be one prediction per test subject
## The first input argument to the script should contain the ground-truth, and the second one the predicted scores
## Both files should have two columns. The first one should be the subject ID and the second one should be 'fluid.resid' for the ground-truth file; and 'predicted_score' for the predicted file

args <- commandArgs(TRUE)
gt_file <- args[1]
pred_file <- args[2]

# ROI 1
pred1 = read.csv("pred_validation/20190324_pred_validation_ROI71_epoch3.csv",header=TRUE) %>%
        dplyr::rename(predicted_score_ROI71 = predicted_score)

# ROI 2
pred2 = read.csv("pred_validation/20190324_pred_validation_ROI77_epoch3.csv",header=TRUE) %>%
        dplyr::rename(predicted_score_ROI77 = predicted_score)

# ROI 3
pred3 = read.csv("pred_validation/20190324_pred_validation_ROI13_epoch3.csv",header=TRUE) %>%
        dplyr::rename(predicted_score_ROI13 = predicted_score)

pred12 <- left_join(pred1, pred2)
pred123 <- left_join(pred12, pred3) %>%
        select(-X) %>%
        transform(predicted_score_median = apply(pred123[,-1], 1, median))

pred <- select(pred1234, subject, predicted_score = predicted_score_median)

fluid_resid_test = read.csv("gt_validation.csv",header=TRUE)
test = merge(fluid_resid_test,pred,by=names(fluid_resid_test)[1])

## Validation MSE
## Only compute MSE and R-squared if the number of predictions made is at least 99% of the test sample
if(dim(test)[1] >= .99*dim(fluid_resid_test)[1]){
	test$predicted_score[is.na(test$predicted_score)] = 
		test$predicted_score[abs(test$predicted_score-test$fluid.resid) == max(abs(test$predicted_score-test$fluid.resid),na.rm=TRUE)]
	r.squared = cor(test$fluid.resid,test$predicted_score)^2
	mse = mean((test$fluid.resid - test$predicted_score)^2)
	# mse = mean((test$fluid.resid - mean(test$fluid.resid))^2)
	cat("MSE: ", mse, "\nR-Squared: ", r.squared, "\n")
}

## Prediction Scores
test_pred1_1 <- read.csv("20190324_pred_testset_ROI13_epoch3_part1.csv") %>%
        select(-X)
test_pred1_2 <- read.csv("20190324_pred_testset_ROI13_epoch3_part2.csv") %>%
        select(-X)
test_pred1 <- merge(test_pred1_1, test_pred1_2, all = T) %>%
        dplyr::rename(predicted_score_ROI13 = predicted_score)

test_pred2_1 <- read.csv("20190324_pred_testset_ROI71_epoch3_part1.csv") %>%
        select(-X)
test_pred2_2 <- read.csv("20190324_pred_testset_ROI71_epoch3_part2.csv") %>%
        select(-X)
test_pred2 <- merge(test_pred2_1, test_pred2_2, all = T) %>%
        dplyr::rename(predicted_score_ROI71 = predicted_score)

test_pred3_1 <- read.csv("20190324_pred_testset_ROI77_epoch3_part1.csv") %>%
        select(-X)
test_pred3_2 <- read.csv("20190324_pred_testset_ROI77_epoch3_part2.csv") %>%
        select(-X)
test_pred3 <- merge(test_pred3_1, test_pred3_2, all = T) %>%
        dplyr::rename(predicted_score_ROI77 = predicted_score)

test_pred12 <- left_join(test_pred1, test_pred2)
test_pred123 <- left_join(test_pred12, test_pred3)
test_pred123 <- transform(test_pred123, predicted_score_median = apply(test_pred123[,-1], 1, median))

test_pred <- select(test_pred123, subject, predicted_score = predicted_score_median)
test_pred

write.csv(test_pred, "test_results.csv", row.names = F)

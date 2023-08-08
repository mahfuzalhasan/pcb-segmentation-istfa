easy_data_folder = '/local/shared/batch-6-data/datagen/Train Validate Test Split'
data_folder = '/local/shared/batch-6-data/datagen/Png Export'
file_name = 'filtered_summary.csv'
data_path = './data'
saved_models_dir = './Results/saved_models'
output_dir = './Results/output'
logs_dir = './Results/logs'
split_set = "filtered_set"
# split_set = "easy_set"

#test_model_path = "/home/UFAD/mdmahfuzalhasan/Documents/Results/PCB_Component_Segmentation/saved_models/09-01-22_0003/model_94.pth"
#test_model_path = "/home/UFAD/mdmahfuzalhasan/Documents/Results/PCB_Component_Segmentation/saved_models/10-17-22_1847/model_56.pth"
#test_model_path = "/home/UFAD/mdmahfuzalhasan/Documents/Results/PCB_Component_Segmentation/saved_models/11-17-22_0213/model_30.pth"
#test_model_path = "./Results/saved_models/12-09-22_0212/model_40.pth"
#test_model_path = "./Results/saved_models/01-22-23_1614/model_62.pth"
#test_model_path = "./Results/saved_models/01-25-23_1335/model_62.pth"

#test_model_path = "./Results/saved_models/02-05-23_1343/model_33.pth"
#test_model_path = "./Results/saved_models/02-07-23_1227/model_42.pth"
#test_model_path = "./Results/saved_models/02-07-23_1559/model_37.pth"
# test_model_path = "./Results/saved_models/02-03-23_1910/model_69.pth" #0%
# test_model_path = "./Results/saved_models/03-06-23_0010/model_99.pth" #0%


###### ResNet 18 and Tversky Loss
# test_model_path = "./Results/saved_models/03-07-23_0145/model_124.pth" #10%
# test_model_path = "./Results/saved_models/03-10-23_0302/model_83.pth" #30%


##### DeepLabv3 with ResNet18 d = 3 5 7 and PatchTversky Loss
# test_model_path = "./Results/saved_models/03-16-23_0159/model_93.pth" # 0%
# test_model_path = "./Results/saved_models/04-20-23_1614/model_144.pth" # 10% 
# test_model_path = "./Results/saved_models/04-21-23_0140/model_67.pth" # 30% 
# test_model_path = "./Results/saved_models/04-21-23_0944/model_37.pth" # 30% 

##### DeepLabv3 with ResNet34 pre-trained and Tversky Loss on old dataset
# test_model_path = "./Results/saved_models/03-26-23_1646/model_25.pth" # 0%
# test_model_path = "./Results/saved_models/03-29-23_1759/model_63.pth" # 0%

## Resuming from 0% training. Starting training with 20% bbox
# resume_model = "./Results/saved_models/07-27-23_2229/model_42.pth"

## Testing CL after Starting training with 20% bbox
# test_model_path = "./Results/saved_models/07-28-23_0308_resume_2229/model_175.pth"
# test_model_path = "./Results/saved_models/07-28-23_0308_resume_2229/model_126.pth"
# test_model_path = "./Results/saved_models/07-28-23_0308_resume_2229/model_119.pth"
# test_model_path = "./Results/saved_models/07-28-23_0308_resume_2229/model_62.pth"
# test_model_path = "./Results/saved_models/07-28-23_0308_resume_2229/model_162.pth"


## Testing CL after Starting training with 40% bbox
# test_model_path = "./Results/saved_models/07-28-23_1322_resume_2229/model_153.pth"

## Testing CL after Starting training with 60% bbox
test_model_path = "./Results/saved_models/08-04-23_1853_resume_2229/model_82.pth"

## Resuming from 20% training. Starting training with 40% bbox
# resume_model = "./Results/saved_models/07-28-23_0308_resume_2229/model_175.pth"

## Resuming from 40% training. Starting training with 60% bbox
resume_model = "./Results/saved_models/07-28-23_1322_resume_2229/model_153.pth"

train_bbox_id = ['pcb_6f','pcb_1f','pcb_10b','pcb_3f','pcb_6b']
test_to_val = ["pcb_74b","pcb_177f","pcb_71f"]
val_to_test = ["pcb_117f", "pcb_40f"]
#train_to_test = ["pcb_47f","pcb_176f"]
train_to_test = ["pcb_172f", "pcb_81f", "pcb_150f", "pcb_19f", "pcb_24f"]
test_to_train = ["pcb_169b", "pcb_114b","pcb_119b","pcb_118f","pcb_112f","pcb_64b","pcb_4b","pcb_93f","pcb_150b","pcb_71b","pcb_152b"]
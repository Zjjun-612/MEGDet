from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
model_cfg = '/workspace/yolo11/ultralytics/cfg/models/11/yolo11-obb-fusion dwt-fly.yaml'
model = YOLO(model_cfg, task='obb')

# Train the model  
results = model.train(data='DroneVehicle_merge.yaml', 
                      epochs=200, 
                      imgsz=640, 
                      batch=16, #m-20,s-32,l-12
                      device='1', ## (int | str | list, optional) device to run on, i.e. cuda device=0 or device='0,1' or device=cpu
                      optimizer='SGD', # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                      # close_mosaic=5, # (int) disable mosaic augmentation for final epochs (0 to disable)
                      project='./runs/test', # (str, optional) project name
                      name='yolov11n_fusion', # (str, optional) experiment name, results saved to 'project/name' directory
                      exist_ok=False, # (bool) whether to overwrite existing experiment
                      #amp= True,

                      # #model='yolov8m-obb.yaml'# (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
                      # #pretrained='./weights/yolov8m.pt', # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
                      # patience=50, # (int) epochs to wait for no observable improvement for early stopping of training
                      # #cache=False, # (bool) True/ram, disk or False. Use cache for data loading
                      # #save=True, # (bool) save train checkpoints and predict results
                      # #save_period=-1, # (int) Save checkpoint every x epochs (disabled if < 1)
                      # cos_lr=True, # (bool) use cosine learning rate scheduler
                      # resume=False, # (bool) resume training from last checkpoint
                      # freeze=None, # (int | list, optional) freeze first n layers, or freeze list of layer indices during training
                      # #multi_scale=False,
                      # lr0=0.0086, # (float) initiallng rate (i.e. SGD=1E-2, Adam=1E-3)
                      # lrf=0.01, # (float) final learning rate (lr0 * lrf)
                      # mosaic=0.5, # (float) image mosaic (probability)
                      # warmup_epochs=3.0, # (float) warmup epochs (fractions ok)
                      # label_smoothing=0.1, # (float) labels smoothing (fraction)
                      # degrees=180, # (float) image rotation (+/- deg)
                      # translate=0.1, # (float) image translation (+/- fraction)
                      # mixup=0, # (float) image mixup (probability)
                      # conf=0.1,
                      # visualize=True
                      )
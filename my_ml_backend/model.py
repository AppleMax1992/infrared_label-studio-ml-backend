from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
import os
from PIL import Image

HOME = os.getcwd()
print(HOME)

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}
        # Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        print(tasks)
        # /data/upload/5/7dc29ed0-INFRARED_Arrester_220kV_00.jpg.
        #  + '/data/upload/5/7dc29ed0-INFRARED_Arrester_220kV_00.jpg'.split('/')[1:]
        path = tasks[0]['data']['image'].split('/')[2:]
        path = f'{HOME}/my-label-studio/mydata/media/' + '/'.join(path)
        # print('aaaaaaaaaaaaaaaaaaaa',path)
        # path = self.get_local_path(path, task_id=tasks[0]['id'])
        # print('bbbbbbbbbbbbbbb',path)
        im = Image.open(path) 
        image_width, image_height = im.size
        model = YOLO(f'{HOME}/my_ml_backend/yolov8s-seg.pt')
        results = model.predict(source=path, conf=0.25)

        img_results = []
        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xywhn.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        # Iterate through the results
        i = 0
        for box, cls, conf in zip(boxes, classes, confidences):
            x, y, w, h = box
            confidence = conf
            detected_class = cls
            name = names[int(cls)]
            print(x, y, w, h,confidence,detected_class,name,image_width,image_height)
            
            i = i + 1
            img_results.append(
                {
                "id": f"result{i}",
                "type": "rectanglelabels",        
                "from_name": "label", "to_name": "image",
                "original_width": image_width, "original_height": image_height,
                "image_rotation": 0,
                "value": {
                "rotation": 0,          
                "x":(x-w/2)*100, "y": (y-h/2)*100,
                "width": w*100, "height": h*100,
                "rectanglelabels": [model.names[int(cls)]]
                }
            })
        print('!!!!!!!!!!!!!!',img_results)

        # 长方形标注格式 
        # results = [
        #             {
        #                 "id": "result1",
        #                 "type": "rectanglelabels",        
        #                 "from_name": "label", "to_name": "image",
        #                 "original_width": 600, "original_height": 403,
        #                 "image_rotation": 0,
        #                 "value": {
        #                 "rotation": 0,          
        #                 "x": 4.98, "y": 12.82,
        #                 "width": 32.52, "height": 44.91,
        #                 "rectanglelabels": ["Airplane"]
        #                 }
        #             },
        #             {
        #                 "id": "result2",
        #                 "type": "rectanglelabels",        
        #                 "from_name": "label", "to_name": "image",
        #                 "original_width": 600, "original_height": 403,
        #                 "image_rotation": 0,
        #                 "value": {
        #                 "rotation": 0,          
        #                 "x": 75.47, "y": 82.33,
        #                 "width": 5.74, "height": 7.40,
        #                 "rectanglelabels": ["Car"]
        #                 }
        #             },
        #             {
        #                 "id": "result3",
        #                 "type": "choices",
        #                 "from_name": "choice", "to_name": "image",
        #                 "value": {
        #                     "choices": ["Airbus"]
        #                 }
        #             }
        #         ]
        
        print(img_results)
        
        # {
        #   "original_width": 800,
        #   "original_height": 450,
        #   "image_rotation": 0,
        #   "value": {
        #     "points": [
        #       [
        #         20.93,
        #         28.90
        #       ],
        #       [
        #         25.86,
        #         64.69
        #       ],
        #       [
        #         38.40,
        #         62.79
        #       ],
        #       [
        #         34.13,
        #         27.48
        #       ]
        #     ]
        #   },
        #   "id": "GHI",
        #   "from_name": "polygon",
        #   "to_name": "image",
        #   "type": "polygon"
        # },
        # return ModelResponse(predictions=[{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": results
        # }])
        # print('我是预测结果',results)
        return ModelResponse(predictions=[{
            "model_version": self.get("model_version"),
            # # score is used for active learning sampling mode
            "score": 0.5,
            "result": img_results
        }])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')


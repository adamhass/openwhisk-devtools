# Adapted from:
# https://github.com/apache/flink-statefun-playground/blob/main/python/showcase/showcase/__main__.py
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import statefun
from faas_utils import decode_request, build_response
from keras.models import load_model
# from aiohttp import web
import my_types
from helpers import reshape_image, decode_netout, do_nms
import logging
import time
logging.basicConfig(level=logging.DEBUG)

detection_topic = 'detections'
functions = statefun.StatefulFunctions()

# experiment parameter
num_cameras = 7

# YOLO config:
model_path = 'resources/model.h5'  # Use a local file
net_h, net_w = 416, 416
shape = (net_h, net_w)
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90,  156, 198,  373, 326],  [
    30, 61, 62, 45,  59, 119], [10, 13,  16, 30,  33, 23]]

# "local state"
# model = load_model(model_path)

batcher_invoked_count = 0
#
# ============ Receives from kafka, runs YOLOv3, sends prediction to Kafka ============
#


@functions.bind("inference/yolov3", specs=[])
async def serving(context: statefun.Context, message: statefun.Message):
    # if len(model) == 0:
    #    print("loading model into memory...")
    #    model.append() # Could be from remote storage
    triggered_time = time.time_ns()
    (image, timestamp, camera) = message.as_type(my_types.IMAGE)
    print("DETECTING ts:", timestamp, "camera:", camera)
    # Do the prediction, ensure correct shape.
    image = reshape_image(image, shape)
    global model
    prediction = model.predict(image)

    # Extract the required data from the prediction and filter the output
    boxes = []
    for i in range(len(prediction)):
        # decode the output of the network
        boxes += decode_netout(prediction[i][0], anchors[i],
                               obj_thresh, nms_thresh, net_h, net_w)
    do_nms(boxes, nms_thresh)
    filtered = filter(lambda box: box.classes[0] > obj_thresh, boxes)
    result = list(filtered)
    detected_time = time.time_ns()
    context.send(
        statefun.message_builder(target_typename='app/batcher',
                                 target_id="0",
                                 value_type=my_types.PREDICTION,
                                 value=(result, timestamp, camera, now1, now2)))

#stored_detections = {}
# ============ 2. Batcher: Receives detections, puts them into batches ============


@functions.bind("app/batcher", specs=[statefun.ValueSpec(name='detections', type=my_types.BATCHES)])
def batch(context: statefun.Context, message: statefun.Message):
    (new_detections, timestamp, camera, triggered_time,
     detected_time) = message.as_type(my_types.PREDICTION)
    global batcher_invoked_count

    detections = context.storage.detections
    if not detections:
        detections = {}
    batcher_invoked_count += 1
    if batcher_invoked_count % 100 == 0:
        print(batcher_invoked_count)

    if timestamp in detections:
        batch = detections.pop(timestamp)
    else:
        batch = []

    batch.append((new_detections, triggered_time, detected_time))
    detections[timestamp] = batch
    # Output all earliest batches if possible
    earliest_batch = min(detections.keys())
    while len(detections[earliest_batch]) > 6:
        batch_complete_time = time.time_ns()
        complete_batch = detections.pop(earliest_batch)
        times = []
        print("outputting batch ", earliest_batch, " to kafka:")
        for (detection, t1, t2) in complete_batch:
            times.append(t1)
            times.append(t2)
        times.append(batch_complete_time)
        context.send_egress(
            statefun.kafka_egress_message(typename=my_types.PREDICTION.typename, topic=detection_topic, key="0",
                                          value_type=my_types.PREDICTION, value=times))
        if len(detections) > 0:
            earliest_batch = min(detections.keys())
        else:
            context.storage.detections = detections
            return
    context.storage.detections = detections

# ============ 3. Aggregator: Receives batches, aggregates them ============


@functions.bind("app/aggregation")
def messaging(context: statefun.Context, message: statefun.Message):
    (prediction_batch, timestamp) = message.as_type(my_types.PREDICTION)
    context.send_egress(
        statefun.kafka_egress_message(typename=my_types.PREDICTION.typename, topic=detection_topic, key="0",
                                      value_type=my_types.PREDICTION, value=prediction_batch))


#
# Serve the endpoint
#
################################################################################
# Static Handler:
# handler = statefun.RequestReplyHandler(functions)
#
#
# async def handle(request):
#     req = await request.read()
#     res = await handler.handle_async(req)
#     return web.Response(body=res, content_type="application/octet-stream")
#
# app = web.Application(client_max_size=10**9)
# logging.basicConfig(level=logging.DEBUG)
# app.add_routes([web.post('/statefun', handle)])
#
# if __name__ == '__main__':
#     print("Serving on port 8000")
#     web.run_app(app, port=8000)
################################################################################
# OpenWhisk Handler:
handler = statefun.RequestReplyHandler(functions)

print("OUTSIDE main()")


def main(args):
    print("INSIDE main()")
    return {"a": "b"}


def faas_handler(request, context):
    message_bytes = decode_request(request)
    response_bytes = handler(message_bytes)
    return build_response(response_bytes)

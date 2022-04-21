#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import threading
import flask
import redis
import json

from werkzeug.serving import WSGIRequestHandler
from flask import jsonify,json,Response

__author__ = 'Benjamin Milde'

redis_server_channel = 'subtitle2go'

app = flask.Flask(__name__)
red = redis.StrictRedis(charset="utf-8", decode_responses=True)

long_poll_timeout = 0.5
long_poll_timeout_burst = 0.08

current_jobs = {}

def persistence_event_stream():
    global current_jobs
    print('Estabilishing persistence event_stream...')
    pubsub = red.pubsub()
    pubsub.subscribe(redis_server_channel)
    for message in pubsub.listen():
        msg = str(message['data'])
        print('New msg:', msg)
        if 'pid' in msg:
            msg_json = json.loads(msg)
            key = str(msg_json['pid']) + '_' + msg_json['file_id']
            current_jobs[key] = msg_json

def event_stream():
    print('New connection to event_stream!')
    pubsub = red.pubsub()
    pubsub.subscribe(reids_server_channel)
 #   yield b'hello'
    for message in pubsub.listen():
        if not message['type'] == 'subscribe':
            #print('New message:', message)
            #print(type(message['data']))
            yield b'data: %s\n\n' % message['data']

@app.route('/status')
def status():
    return jsonify(current_jobs)

@app.route('/status/<jobid>')
def status(jobid):
    if jobid in current_jobs:
        return jsonify(current_jobs[jobid])
    else:
        return jsonify({"error":"could not find jobid in current jobs."})

@app.route('/clear')
def clear_finished():
    to_delete = []
    for key in current_jobs:
        if 'finished' in current_jobs[key]['status'] \
          or 'failed' in current_jobs[key]['status'] :
            to_delete.append(key)
    for key in to_delete:
        del current_jobs[key]
    return 'ok'

#Event stream end point for the browser, connection is left open. Must be used with threaded Flask.
@app.route('/stream')
def stream():
    return flask.Response(event_stream(), mimetype='text/event-stream')

#Traditional long polling. This is the fall back, if a browser does not support server side events. 
@app.route('/stream_poll')
def poll():
    pubsub = red.pubsub()
    pubsub.subscribe(redis_server_channel)
    message = pubsub.get_message(timeout=long_poll_timeout)
    while(message != None):
        yield message
        message = pubsub.get_message(timeout=long_poll_timeout_burst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Status update server for subtitle2go jobs')
    parser.add_argument('-l', '--listen-host', default='127.0.0.1', dest='host', help='Host address to listen on.')
    parser.add_argument('-p', '--port', default=7500, dest='port', help='Port to listen on.', type=int)
    parser.add_argument('--debug', dest='debug', help='Start with debugging enabled', action='store_true', default=False)

    args = parser.parse_args()

    #print(' * Starting app with base path:',base_path)
    if args.debug:
        app.debug = True
 
    persistence_event_stream_thread = threading.Thread(target=persistence_event_stream)
    persistence_event_stream_thread.start()
    print('Running as testing server.')
    print('Host:', args.host)
    print('Port:', args.port)

    WSGIRequestHandler.protocol_version = 'HTTP/1.1'
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False, use_debugger=False)

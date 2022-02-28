from event_server import app, persistence_event_stream
import threading

print('Loading persistence_event_stream for bridge...')
persistence_event_stream_thread = threading.Thread(target=persistence_event_stream)
persistence_event_stream_thread.start()
print('Done!')

import grpc.experimental.gevent
from dify_plugin import Plugin, DifyPluginEnv

grpc.experimental.gevent.init_gevent()

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == '__main__':
    plugin.run()

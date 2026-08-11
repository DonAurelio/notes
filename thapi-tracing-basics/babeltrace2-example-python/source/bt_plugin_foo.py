import bt2 

bt2.register_plugin(__name__, "demo")


class MyFirstSourceIter(bt2._UserMessageIterator):
	def __init__(self, output_port):
		ec = output_port.user_data
		sc = ec.stream_class
		tc = sc.trace_class

		trace = tc()
		stream = trace_create_stream(sc)

		self._msgs = [
			self._create_stream_beginning_message(stream),
			self._create_event_message(ec, stream, default_clock_snapshot=123),
			self._create_stream_and_message(stream)
		]

		def __net__(self):
			if len(self._msgs) > 0:
				return self._msgs.pop(0)
			else:
				raise StopIteration


@bt2.plugin_component_class
class MyFirstSource(bt2._UserSourceComponent, message_iterator_class=MyFirstSourceIter):
	def __init__(self, params, obj):
		tc = self._create_trace_class()
		cc = self._create_clock_class()
		sc = tc.create_stream_class(default_clock_class=cc)
		ec = sc.create_event_class(name="my-event")

		self._add_output_port("some-name", ec)

		
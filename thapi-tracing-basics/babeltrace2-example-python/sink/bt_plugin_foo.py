"""
HOW TO RUN THIS FROM THE COMMAND LINE

babeltrace2 --plugin-path=. -c sink.demo.MyFirstSink /home/aureavm/THAPI/babeltrace2-example/sink_output

--plugin-path: adds the path to the components
--component (-c): points to the component and the class
<PATH TO LTTNG TRACES>: The path to lttng traces
"""

import bt2 

bt2.register_plugin(__name__,"demo")

"""
@bt2.plugin_component_class tells babeltrace the 
following class is a component class.
"""
@bt2.plugin_component_class
class MyFirstSink(bt2._UserSinkComponent):
	"""MyFirstSink Docs

	bt2._UserSinkComponent indicate the component is 
	created by the user.
	"""

	def __init__(self, params, obj, other):
		"""
		
			Args:
				params: parametes pased to configure the component 
					upon creation. Parameters can be passed from the 
					application of the command line.
				obj: If the application is being developed in Python 
					you can pass an object to the component that will 
					be instantiated, this object can be a db connection 
					or something needed in the component from the outside
					application.

		"""

		# input ports are created in the __init__ method (callback).
		self._port = self._add_input_port("some_name")

	def _user_graph_is_configured(self):

		# Messages iterators are created in the _user_graph_is_configured method (callback).
		self._it = self._create_message_iterator(self._port)

	def _user_consume(self):

		# In consume you will be able to do something useful with your message.
		msg = next(self._it)
		print(msg)


""""
* Four streams begin in parallel 
- Are these related to the number of cores in my machine?
- How we can control these?
<bt2.message._StreamBeginningMessageConst object @ 0x563a2875a150>
<bt2.message._StreamBeginningMessageConst object @ 0x563a2875bd60>
<bt2.message._StreamBeginningMessageConst object @ 0x563a2875c320>
<bt2.message._StreamBeginningMessageConst object @ 0x563a2875cb50>

- What do these packages mean?
<bt2.message._PacketBeginningMessageConst object @ 0x563a2875a210>
<bt2.message._PacketBeginningMessageConst object @ 0x563a2875be20>
<bt2.message._PacketBeginningMessageConst object @ 0x563a2875c3e0>
<bt2.message._PacketBeginningMessageConst object @ 0x563a2875cc10>

* The three events obtained from the lttng-example 
<bt2.message._EventMessageConst object @ 0x563a2875a4f0>
<bt2.message._EventMessageConst object @ 0x563a2875a770>
<bt2.message._EventMessageConst object @ 0x563a2875b1f0>

* Packages and streams close
<bt2.message._PacketEndMessageConst object @ 0x563a2875b2b0>
<bt2.message._StreamEndMessageConst object @ 0x563a2875ba50>

<bt2.message._PacketEndMessageConst object @ 0x563a2875bee0>
<bt2.message._StreamEndMessageConst object @ 0x563a2875bfa0>

<bt2.message._PacketEndMessageConst object @ 0x563a2875c4a0>
<bt2.message._StreamEndMessageConst object @ 0x563a2875c5f0>

<bt2.message._PacketEndMessageConst object @ 0x563a2875ccd0>
<bt2.message._StreamEndMessageConst object @ 0x563a2875cd90>
"""

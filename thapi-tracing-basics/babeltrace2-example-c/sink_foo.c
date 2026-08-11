#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <babeltrace2/babeltrace.h>


/*
HOW TO RUN THIS CODE
$ cc sink_foo.c -fPIC -c $(pkg-config --cflags babeltrace2)
$ ld sink_foo.o -o sink_foo.so -shared $(pkg-config --libs babeltrace2)

$ babeltrace2 --plugin-path=. /path/to/ctf/trace --component=sink.foo.output
*/


/* Sink component's private data 

The name of the plugin is "foo" and the name of the 
componente class is "output". Therefore, the component 
class is identified in the babeltrace2 command-line tool 
as "sink.foo.output".

A sink.foo.output component prints one text line to the 
standard output for each event message it consumes, for 
example:

#1: kmem_kmalloc (5 payload members)
#2: kmem_kfree (2 payload members)
#3: sched_waking (4 payload members)
#4: sched_migrate_task (5 payload members)
#5: sched_stat_runtime (4 payload members)
#6: sched_wakeup (4 payload members)
#7: rcu_utilization (1 payload member)
#8: rcu_utilization (1 payload member)
#9: sched_switch (7 payload members)
#10: syscall_entry_write (3 payload members)

For each line, there is:
- The event message's index (simple counter).
- The event message's event class name.
- The number of members in the event message's event's playload field.

A sink.foo.output component does not need any initialization parameter:
it just prints to the standard output.

A sink.foo.output component creates a single input port named "in".

To simplify this example, a sink.foo.output component does not check 
the return status codes of API functions, but you must check them in 
production code.

The sink component class implemnattion and the shared object plugin macros 
are in the same file, bt_plugin_foo.c:

*/


/* Sink components private data */
struct Foo 
{
	/* Uptream message iterator (owned by this) */
	bt_message_iterator *message_iterator;

	/* Current event message index */
	uint64_t index;
};


/**
 * Initializes the sink component.
 */
static 
bt_component_class_initialize_method_status 
foo_initialize(
	bt_self_component_sink *self_component_sink,
	bt_self_component_sink_configuration *configuration,
	const bt_value *params,
	void *initialize_method_data
)
{
	/* Allocate a private data structure */
	struct Foo *foo = malloc(sizeof(struct Foo)); 

	/* Initialize the firs event message's index */
	foo->index = 1;

	/* Set the component's user data to our private data structure */
	bt_self_component_set_data(
		bt_self_component_sink_as_self_component(self_component_sink),
		foo
		);

	/**
	 * Add and input port named "in" to the sink component.
	 * 
	 * This is needed so that this sink component can be 
	 * connected to a filter or a source component. With
	 * a connected upstream component, this sink component 
	 * can create a message iterator to consume messages. 
	 *  
	 */

	bt_self_component_sink_add_input_port(
		self_component_sink, "in", NULL, NULL
		);

	return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_OK;	
}


/**
 * Finalizes the sink component. 
 */

static 
void foo_finalize(bt_self_component_sink *self_component_sink)
{

	/* Retrieve our private data from the component's user data */
	struct Foo *foo = bt_self_component_get_data(
		bt_self_component_sink_as_self_component(self_component_sink));

	/* Free the allocated structure */
	free(foo);
}


/**
 * Called when the trace processing graph containing the sink
 * component is configured.
 * 
 * This is where we can create our message iterator.
 */

static
bt_component_class_sink_graph_is_configured_method_status
foo_graph_is_configured(
	bt_self_component_sink *self_component_sink
)
{
	/* Retrieve our private data from the component's user data */
	struct Foo *foo = bt_self_component_get_data(
		bt_self_component_sink_as_self_component(self_component_sink));

	/* Borrow our unique port */
	bt_self_component_port_input *in_port = 
		bt_self_component_sink_borrow_input_port_by_index(
			self_component_sink, 0);

	/* Create the upstream message iterator */
	// ERROR? &foo is not correct but foo->message_iterator
	bt_message_iterator_create_from_sink_component(
		self_component_sink, in_port, &foo->message_iterator);
	
	return BT_COMPONENT_CLASS_SINK_GRAPH_IS_CONFIGURED_METHOD_STATUS_OK;	
}


/**
 * Prints a line of "message", if it's an event message,
 * to the standard output.
 */
static 
void print_message(struct Foo *foo, const bt_message *message)
{
	/* Discard if it's not an event message */
	if (bt_message_get_type(message) != BT_MESSAGE_TYPE_EVENT)
	{
		goto end;
	}

	/* Borrow the event message's, event and its class */
	const bt_event *event = 
		bt_message_event_borrow_event_const(message);

	const bt_event_class *event_class = 
		bt_event_borrow_class_const(event);

	/* Get the number of payload field members */
	const bt_field *payload_field = 
		bt_event_borrow_payload_field_const(event);
	uint64_t member_count = bt_field_class_structure_get_member_count(
		bt_field_borrow_class_const(payload_field));

	/* Write a corresponding line to the standar output */
	// ERROR? PRIu64 defined ? where ?
	printf("#%" PRIu64 ": %s (%" PRIu64 " payload members%s)\n",
		foo->index, bt_event_class_get_name(event_class),
		member_count, member_count == 1 ? "" : "s");

	/* Increment the current event message's index */
	foo->index++;

end:
	return;
}

/**
 * Consumes a batch of messages and writes the corresponding lines
 * to the standard output.
 */
bt_component_class_sink_consume_method_status 
foo_consume(
	bt_self_component_sink *self_component_sink)
{
	bt_component_class_sink_consume_method_status status = 
		BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_OK;

	/* Retrieve our private data from the component's user data */
	struct Foo *foo = bt_self_component_get_data(
		bt_self_component_sink_as_self_component(self_component_sink));

	/* Consume a batch of messages from the upstream message iterator */
	bt_message_array_const messages;
	uint64_t message_count;
	bt_message_iterator_next_status next_status =
		bt_message_iterator_next(
			foo->message_iterator, &messages, &message_count);

	switch (next_status) 
	{

		case BT_MESSAGE_ITERATOR_NEXT_STATUS_END:
			/* End of iteration: put the message iterator's reference */
			bt_message_iterator_put_ref(foo->message_iterator);
			status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_END;
			goto end;

		case BT_MESSAGE_ITERATOR_NEXT_STATUS_AGAIN:
			status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_AGAIN; 
			goto end;

		case BT_MESSAGE_ITERATOR_NEXT_STATUS_MEMORY_ERROR:
			status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_MEMORY_ERROR;
			goto end;

		case BT_MESSAGE_ITERATOR_NEXT_STATUS_ERROR:
			status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_ERROR;
			goto end;
		default:
			break;
	}

	/* For each consumed message */
	for (uint64_t i = 0; i < message_count; i++)
	{
		/* Current message */
		const bt_message *message = messages[i];

		/* Print line for current message if it's and event message */
		print_message(foo, message);

		/* Put this message's reference */
		bt_message_put_ref(message);

	}

end: 
	return status;

} 


/* Mandatory */
BT_PLUGIN_MODULE();
 
/* Define the `foo` plugin */
BT_PLUGIN(foo);
 
/* Define the `foo` sink component class */
BT_PLUGIN_SINK_COMPONENT_CLASS(output, foo_consume);
 
/* Set some of the `output` sink component class's optional methods */
BT_PLUGIN_SINK_COMPONENT_CLASS_INITIALIZE_METHOD(output, foo_initialize);
BT_PLUGIN_SINK_COMPONENT_CLASS_FINALIZE_METHOD(output, foo_finalize);
BT_PLUGIN_SINK_COMPONENT_CLASS_GRAPH_IS_CONFIGURED_METHOD(output,foo_graph_is_configured);








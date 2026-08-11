#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdbool.h>
#include <babeltrace2/babeltrace.h>

/*

HOW TO RUN THIS CODE

$ cc distill.c -fPIC -c $(pkg-config --cflags babeltrace2)
$ ld distill.o -o distill.so -shared $(pkg-config --libs babeltrace2)


$ babeltrace2 --plugin-path=. /path/to/ctf/trace \
              --component=filter.distill.theone \
              --params='names=["sched_switch", "rcu_utilization", "kmem_kfree"]'
*/


/*
Simple filter component class

This example shows a basic filter component class packaged as a 
shared object plugin.

The name of the plugin is "bar" and the name of the filter
component class is "muxer". therefore the component class
if defined in the babeltrace2 command-line tool as
"filter.bar.muxer".

A filter.bar.muxer component accepts a single initialization
parameter, "names", which is an array value of string values.
The array values contains the names of the classes of the events 
to discard.

A fiter.bar.muxer component creates a single input port named
"in" and a single output port named out.

To simplify this example, a filter.bar.muxer component is not
resilient and needs a valid input and valid initialization 
parameters. The code also does not check the return status 
code of API functions for simplicity, but you must check them
in production code.

The filter component class implementation and the shared object
plugin macros are in the same file, bar.c:
*/


/* Filter component's private data */
struct bar {
	/* Names of the clases of the evenets to discard (owned by this) */
	const bt_value *names_value;

	/* Component's input port (weak) */
	bt_self_component_port_input *in_port;
};

/**
 * Initializes the filter component. 
 **/
static
bt_component_class_initialize_method_status bar_initialize(
	bt_self_component_filter *self_component_filter,
	bt_self_component_filter_configuration *cofiguration,
	const bt_value *params,
	void *initialize_method_data)
{
	/* Allocate a private data structure */
	struct bar *bar = malloc(sizeof(*bar));

	/**
	 * Keep a reference of the "names" array value parameter so that 
	 * the "next" method of a message iterator can access it to 
	 * decide whether or not to discard and event message.
	 **/
	bar->names_value = 
		bt_value_map_borrow_entry_value_const(params,"names");
	
	/* bt_value_get_ref: increaments the reference count of the value */
	bt_value_get_ref(bar->names_value);

	/* Set the component's user data to our private data structure */
	/* Upcasts the self filter component to the common bt_self_component type*/
	bt_self_component_set_data(
		bt_self_component_filter_as_self_componet(self_component_filter),bar);

	/**
	 * Add an input port named "in" to the filter component.
	 * 
	 * This is needed so that filter component can be connected to
	 * a filter or a source component. With a connected upstream 
	 * component, this filter component's message iterator can create
	 * a message iterator to consume messages.
	 * 
	 * Add and output port named "out" to the filter component.
	 * 
	 * This is needed so that this filter component can be connected to
	 * a filter or a sink component. Once a downstream component is connected,
	 * it can create our message iterator. 
	 **/

	bt_self_component_filter_add_input_port(
		self_component_filter, "in", NULL, &bar->in_port);
	bt_self_component_filter_add_output_port(
		self_component_filter,"out", NULL, NULL);

	return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_OK;
}


/**
 * Finalizes the filter component 
 **/
static void bar_finalize(
	bt_self_component_filter *self_component_filter)
{

	/* Retrieve our private data from the compoent's user data */
	struct bar *bar = bt_self_component_get_data(
		bt_self_component_filter_as_self_component(self_component_filter));

	/* Put all references */
	/* bt_value_put_ref: Decrements the reference count of a value */
	bt_value_put_ref(bar->names_value);

	/* Free the allocated structure */
	free(bar);
}


/* Message iterator's private data */
struct bar_message_iterator {
	/* (Weak) link to the component's private data */
	struct bar *bar;

	/* Upstream message iterator (owned by this) */
	bt_message_iterator *message_iterator;
}

/**
 * Initializes the message iterator
 **/
static
bt_message_iterator_class_initialize_method_status
bar_message_iterator_initialize(
	bt_self_message_iterator *self_message_iterator,
	bt_self_message_iterator_configuration *configuration,
	bt_self_component_port_output *self_port)
{
	/* Allocate a private data structure */
	struct bar_message_iteartor *bar_iter = 
		malloc(sizeof(*bar_iter));

	/* Retrieve the component's private data from its user data */
	struct bar *bar = bt_self_component_get_data(
		bt_self_message_iteartor_borrow_componet(self_message_iterator));

	/* Keep a link to the component's private data */
	bar_iter->bar = bar;

	/* Create the upstream message iterator */
	bt_message_iteartor_create_from_message_iteartor(
		self_message_iteartor,
		bar->in_port, &bar_iter->message_iterator);

	/* Set the message iterator's user data to our private data structure */
	bt_self_message_iterator_set_data(
		self_message_iterator, bar_iter);

	return BT_MESSAGE_ITERATOR_CLASS_INITIALIZE_METHOD_STATUS_OK;

}


/**
 * Finalizes the message iterator.  
 **/
static 
void bar_message_iterator_finalize(
	bt_message_iterator *self_message_iterator)
{
	/* Retrieve our private data from the message iterator's user data */
	struct bar_message_iteartor *bar_iter = 
		bt_self_message_iterator_get_data(self_message_iterator);	

	/* Free the allocated structure */
	free(bar_iter);
}


/**
 * Returns "true" if "message" passes, that is, one of 
 * 
 * - Its not an event message.
 * - The event message does not need to be discarded based on its event class's name.
 */
static 
bool message_passes(
	struct bar_message_iterator *bar_iter, const bt_message *message)
{
	bool passes = true;

	/* Move as is if it's not an event message */
	if(bt_message_get_type(message) != BT_MESSAGE_TYPE_EVENT) {
		passes = false;
		goto end;
	}

	/* Borrow the event message's event and its class */
	const bt_event *event =
		bt_message_event_borrow_event_const(message);
	const bt_event_class *event_class =
		bt_event_borrow_class_const(event);

	/* Get the event class's name */
	const  char *name = bt_event_class_get_name(event_class);

	for (uin64_t i = 0; i < bt_value_array_get_length(bar_iter->bar->names_values); i++)
	{
		const char *discard_name = bt_value_string_get(
			bt_value_array_borrow_element_by_index_const(
				bar_iter));

		if (strcmp(name, discard_name) == 0) {
			passes = false;
			goto end;
		}

	}

end:
	return passes;

}


/**
 * Returns the next message to the message iterator's user.
 * 
 * This method can fill the `messages` array with up to `capacity`
 * messages.
 * 
 * To keep this example simple, we put a single message into 
 * `messages`.
 * 
 * and set `*count` to 1 (if the message iterator is not ended). 
 **/
static
bt_message_iteartor_class_next_method_status
bar_message_iterator_next(
	bt_self_message_itetaror *self_message_iterator,
	bt_message_array_const messages,
	uint64_t capacity,
	uint64_t *count)
{
	/* Retrieve our private data from the message iterator's user data */
	struct bar_message_iterator *bar_iter =
		bt_self_message_iterator_get_data(self_message_iterator);

	/* Consume a batch of messages from the upstream message iterator */
	bt_message_array_const upstream_messages;
	uint64_t upstream_message_count;
	bt_message_iterator next_status;

consume_upstream_messages:
	next_status = 
		bt_message_iterator_next(
			bar_iter->message_iterator,
			&upstream_messages,
			&upstream_message_count);

	/* Initialize the return status to a success */
	bt_message_iterator_class_next_method_status status = 
		BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_OK;

	switch (next_status) {
	
		case BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_END:
			/* End of iteration: put the message iterator's reference */
			bt_message_iterator_put_ref(bar_iter->message_iterator);
			goto end;

		case  BT_MESSAGE_ITERATOR_NEXT_STATUS_AGAIN:
			status =
				BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_AGAIN;
			goto end;

		case BT_MESSAGE_ITERATOR_NEXT_STATUS_MEMORY_ERROR:
			status =
				BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_MEMORY_ERROR;
			goto end;

		case BT_MESSAHE_ITERATOR_NEXT_STATUS_ERROR:
			status =
				BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_ERROR;
			goto end;

		default:
			break;
	}

	/* Output message array index */
	uint64_t i = 0;

	/* For each consumed message */
	for (uint64_t upstream_i = 0; upstream_i < upstream_message_count; upstream_i ++)
	{
		/* Current message */
		const bt_message *upstream_message =
			upstream_messages[upstream_i];

		/* Check if the upstream message passes */
		if (message_passes(bar_iter, upstream_message)){
			/* Move upstream message to output message array */
			messages[i] = upstream_message;
			i++;
			continue;
		}

		/* Discard upstream message put it reference */
		bt_message_put_ref(upstream_message);

	}

	if (i == 0){
		/**
		 * We discarded all the upstream messages: get a new batch
		 * of messages, because this method cannot return
		 * `BT_MESSAGE_ITERATOR_CLASS_NEXT_STATUS_OK` and 
		 * put no messages into its output messages array.
		 **/
		goto consume_upstream_messages;
	}

end:
	return status;
}


/* Mandatory */
BT_PLUGIN_MODULE();
 
/* Define the `bar` plugin */
BT_PLUGIN(bar);
 
/* Define the `muxer` filter component class */
BT_PLUGIN_FILTER_COMPONENT_CLASS(muxer, bar_message_iterator_next);
 
/* Set some of the `muxer` filter component class's optional methods */
BT_PLUGIN_FILTER_COMPONENT_CLASS_INITIALIZE_METHOD(muxer, bar_initialize);
BT_PLUGIN_FILTER_COMPONENT_CLASS_FINALIZE_METHOD(muxer, bar_finalize);
BT_PLUGIN_FILTER_COMPONENT_CLASS_MESSAGE_ITERATOR_CLASS_INITIALIZE_METHOD(muxer, bar_message_iterator_initialize);
BT_PLUGIN_FILTER_COMPONENT_CLASS_MESSAGE_ITERATOR_CLASS_FINALIZE_METHOD(muxer, bar_message_iterator_finalize);










/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/external.hpp
 *
 * License: MIT
 */

#include "app.hpp"

namespace chatty
{
	
std::optional<boost::json::object> prepare_payload(
	ApplicationState& app_state, uint32_t peer_id, uint32_t self_id
);

void send_message_llama(
	ApplicationState& state, const boost::json::object& payload,
	std::shared_ptr<dynamic_to_render> tmp_stream
);

}

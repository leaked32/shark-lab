/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/external.cpp
 *
 * License: MIT
 */

#include "app.hpp"

#include "external.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/json.hpp>
#include <boost/json/array.hpp>
#include <boost/json/object.hpp>
#include <boost/json/parse.hpp>

namespace json = boost::json;


// ==============================================================================================
// LLAMA API
// ==============================================================================================

void chatty::send_message_llama(
	ApplicationState& app_state, const boost::json::object& payload,
	std::shared_ptr<chatty::dynamic_to_render> tmp_stream
) 
{
	// auto payload = this->prepare_payload(peer_id, self_id);
	using namespace chatty;
	
	namespace beast = boost::beast; // from <boost/beast.hpp>
	namespace http = beast::http;   // from <boost/beast/http.hpp>
	namespace net = boost::asio;    // from <boost/asio.hpp>
	
	using tcp = net::ip::tcp; // from <boost/asio/ip/tcp.hpp>
	
	
	auto host = app_state.uni_config_->server_address_;
	auto port = app_state.uni_config_->server_port_;
	
	try {
		const std::string target = "/v1/chat/completions";
		
		// === JSON Payload with streaming ===
		
		std::string json_body = json::serialize(payload);
		// shark::log::debug("payload: {}", json_body);
		
		// === HTTP Connection ===
		net::io_context ioc;
		net::ip::tcp::resolver resolver(ioc);
		beast::tcp_stream stream(ioc);
		
		auto results = resolver.resolve(host, std::to_string(port));
		stream.connect(results);
		
		http::request<http::string_body> req{http::verb::post, target, 11};
		req.set(http::field::host, host);
		req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
		req.set(http::field::content_type, "application/json");
		req.body() = json_body;
		req.prepare_payload();
		
		http::write(stream, req);
		
		// === Read streaming response ===
		beast::flat_buffer buffer;
		http::response_parser<http::string_body> parser;
		parser.body_limit((std::numeric_limits<std::uint64_t>::max)());
		
		std::string full_reply;
		std::string pending;
		
		while (!app_state.shutting_down()) {
			beast::error_code ec;
			
			http::read_some(stream, buffer, parser, ec);
			
			if (ec == http::error::end_of_stream)
				break;
			
			if (ec)
				throw beast::system_error(ec);
			
			auto &msg = parser.get();
			
			pending += msg.body();
			msg.body().clear();
			
			size_t line_end;
			
			while ((line_end = pending.find('\n')) != std::string::npos) {
				std::string line = pending.substr(0, line_end);
				pending.erase(0, line_end + 1);
				
				if (!line.starts_with("data: "))
					continue;
				
				std::string data = line.substr(6);
				
				data.erase(0, data.find_first_not_of(" \t\r\n"));
				
				if (data == "[DONE]")
					goto done;
				
				try {
					auto obj = json::parse(data).as_object();
					auto choices = obj["choices"].as_array();
					
					if (choices.empty())
						continue;
					
					auto delta = choices[0].as_object()["delta"].as_object();
					
					if (!delta.contains("content"))
						continue;
					
					if (!delta["content"].is_string())
						continue;
					
					std::string token = delta["content"].as_string().c_str();
					// std::cout << token << std::flush;
					full_reply += token;
					
					if (tmp_stream) {
						std::scoped_lock _(tmp_stream->tmp_mtx_stream);
						if (tmp_stream->status ==
							dynamic_to_render::status::INTERRUPTED) {
							stream.socket().shutdown(
								tcp::socket::shutdown_both);
							stream.socket().close();
						
						return;
							}
							tmp_stream->tmp_stream += token;
					}
				} catch (const std::exception& e) {
					shark::log::debug("JSON parse failed: {}", e.what());
				}
			}
			if (parser.is_done())
				break;
		}
		
		done:
		stream.socket().shutdown(tcp::socket::shutdown_both);
		stream.socket().close();
		
		if (tmp_stream->on_completed.has_value()) {
			tmp_stream->on_completed.value()(full_reply);
		}
		// this->insert_message(peer_id, self_id, full_reply);
		tmp_stream->status = dynamic_to_render::status::COMPLETED;
	} catch (const std::exception &e) {
		tmp_stream->failed = true;
		shark::raise("Error: {}", e.what());
		// std::cerr << "Error: " << e.what() << std::endl;
	}
}


std::optional<boost::json::object> chatty::prepare_payload(
	ApplicationState& app_state, uint32_t peer_id, uint32_t self_id
) 
{
	constexpr size_t MODEL_CONTEXT = 4096;
	constexpr size_t RESPONSE_BUDGET = 1024;
	constexpr size_t SAFETY_MARGIN = 256;
	constexpr size_t MESSAGE_OVERHEAD = 16;
	
	constexpr size_t CONTEXT_BUDGET = MODEL_CONTEXT - RESPONSE_BUDGET - SAFETY_MARGIN;
	
	auto messages = app_state.uni_db_->get_messages_for_peer(peer_id, self_id);
	auto lore_entries = app_state.uni_db_->get_lorebook_for_peer(peer_id);
	
	std::optional<peer> peer_info_opt = app_state.uni_db_->get_peer_by_id(peer_id);
	if (!peer_info_opt.has_value()) {
		shark::raise("peer does not exist in database: peer_id {}", peer_id);
	}
	
	const peer &peer_info = *peer_info_opt;
	if (peer_info.card.empty()) {
		
		ActivityModalText modal {
			std::format("failed to load character card for peer: peer_id {}", peer_id),
			"Error"
		};
		app_state.states_.emplace_back(
			std::make_unique<ActivityModalText>(std::move(modal))
		);
		shark::log::exception("failed to load character card for peer: peer_id {}", peer_id);
		return std::nullopt;
	}
	
	std::string context_text;
	context_text.reserve(messages.size() * 64);
	
	for (const auto &m : messages) {
		context_text += m.content;
		context_text += '\n';
	}
	
	std::vector<std::string> injected_lore;
	injected_lore.reserve(lore_entries.size());
	
	for (const auto &lore : lore_entries) {
		if (lore.keyword.empty())
			continue;
		
		
		std::string keyword_lower = shark::str::to_lower(lore.keyword);
		if (context_text.find(keyword_lower) != std::string::npos) {
			injected_lore.emplace_back(lore.keyword + ": " + lore.content);
		}
	}
	
	std::string base_system = R"(
		You are participating in a roleplay conversation.

		Stay in character at all times.
		Do not break roleplay.
		Respond as the character in a natural conversational style.
		Never mention prompts, system messages, or instructions.
	)";
	
	std::string system_prompt = base_system;
	
	// system_prompt.reserve(2048);
	system_prompt += std::format("conversation_count={}\n\n", messages.size());
	system_prompt += peer_info.card;
	
	if (!injected_lore.empty()) {
		if (!system_prompt.empty())
			system_prompt += "\n\n";
		
		system_prompt += "[LOREBOOK]\n";
		
		for (const auto &l : injected_lore) {
			system_prompt += "- ";
			system_prompt += l;
			system_prompt += '\n';
		}
	}
	
	if (system_prompt.empty()) {
		system_prompt = "You are not an assistant.";
	}
	
	json::array payload_messages;
	
	payload_messages.emplace_back(
		json::object{{"role", "system"}, {"content", system_prompt}});
	
	std::deque<message> kept;
	
	size_t used = shark::str::estimate_tokens(system_prompt);
	
	for (auto it = messages.rbegin(); it != messages.rend(); ++it)
	{
		size_t cost =
		shark::str::estimate_tokens(it->content) + 16;
		
		if (used + cost > CONTEXT_BUDGET)
			break;
		
		used += cost;
		kept.push_front(*it);
	}
	
	for (const auto &m : kept) {
		payload_messages.emplace_back(json::object{
			{"role", m.sender_id == self_id ? "user" : "assistant"},
			{"content", m.content}});
	}
	
	return json::object{{"model", "llama3.1"},
	{"messages", std::move(payload_messages)},
	{"temperature", 1.2},
	{"max_tokens", 1024},
	{"stream", true}};
}



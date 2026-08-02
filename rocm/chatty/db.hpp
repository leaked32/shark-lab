/*
* Project: shark-lab
* Repository: https://github.com/leaked32/shark-lab
*
* File: chatty/db.hpp
*
* License: MIT
*/

#pragma once

#include <cstdint>
#include <memory>

#include <boost/json/object.hpp>
#include <sqlite3.h>

#include "shark/shark.hpp"

namespace chatty
{

struct peer {
	uint32_t id;
	std::string name;
	std::string card;
};

struct message {
	uint32_t id;
	uint32_t sender_id;
	uint32_t reader_id;
	std::string content;
	std::string created_at;
};


struct lore_row {
	int id;
	std::string keyword;
	std::string content;
};

struct config {
	config(std::string_view path);
	void save() const;
	~config();
	
	uint32_t peer_id_ = 1;
	std::string server_address_ = "127.0.0.1";
	uint32_t server_port_ = 37201;
	
	const std::string path_;
};

class db {
public:
	db(std::string_view path);
	~db();
	void init();

	uint32_t insert_peer(const std::string &name, const std::string &card);
	void update_peer(uint32_t id, const std::string &name, const std::string &card);
	void remove_peer(uint32_t peer_id);
	std::optional<peer> get_peer_by_id(uint32_t peer_id);
	std::vector<peer> get_all_peers();
	
	uint32_t insert_message(uint32_t sender_id, uint32_t reader_id, const std::string &content);
	void remove_last_message(uint32_t peer_id, uint32_t self_id);
	std::vector<message> get_messages_for_peer(uint32_t peer_id, uint32_t self_id);
	
	std::vector<lore_row> get_lorebook_for_peer(uint32_t peer_id);
	void insert_lorebook(uint32_t peer_id, const std::string &keyword, const std::string &content);
	void delete_lorebook_entry(int id);
	void update_lorebook_entry(int id, const std::string &keyword, const std::string &content);

private:
	sqlite3 *db_ = nullptr;
	std::mutex mutex_;
};


struct dynamic_to_render {
	std::mutex tmp_mtx_stream;
	std::string tmp_stream;
	enum class status {
		STREAMING,
		INTERRUPTED,
		COMPLETED
	} status = status::STREAMING;
	std::optional<std::function<void(const std::string& reply)>> on_completed = std::nullopt;
	dynamic_to_render() {};
};

// namespace ends
}



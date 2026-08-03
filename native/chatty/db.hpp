/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/db.hpp
 *
 * License: MIT
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/json/object.hpp>
#include <sqlite3.h>

#include "shark/shark.hpp"

namespace chatty
{

struct peer
{
	uint32_t id;
	std::string name;
	std::string card;
};

struct message
{
	uint32_t id;
	uint32_t sender_id;
	uint32_t reader_id;
	std::string content;
	std::string created_at;
};

struct lore_row
{
	int id;
	std::string keyword;
	std::string content;
};

struct color4
{
	float r;
	float g;
	float b;
	float a;

	template <typename T>
	T get() const
	{
		return {r, g, b, a};
	}
};

struct config
{
	config(std::string_view path);
	void save() const;
	~config();

	uint32_t peer_id_ = 1;
	std::string server_address_ = "127.0.0.1";
	std::string font_ = {};
	uint32_t server_port_ = 37201;
	bool dark_mode_ = false;

	color4 window_bg_{0.12f, 0.05f, 0.10f, 0.92f};
	color4 frame_bg_{0.25f, 0.10f, 0.20f, 0.90f};
	color4 button_{0.75f, 0.25f, 0.55f, 1.0f};
	color4 button_hovered_{0.95f, 0.45f, 0.75f, 1.0f};
	color4 button_active_{1.0f, 0.60f, 0.85f, 1.0f};
	color4 text_{1.f, 1.f, 1.f, 1.f};

	std::string background_path_ = "chatty_background.png";

	const std::string path_;
};

class db
{
  public:
	using peer_list = std::vector<peer>;
	using peer_snapshot = std::shared_ptr<const peer_list>;

	db(std::string_view path);
	~db();
	void init();

	uint32_t insert_peer(const std::string& name, const std::string& card);
	void update_peer(uint32_t id, const std::string& name, const std::string& card);
	void remove_peer(uint32_t peer_id);
	std::optional<peer> get_peer_by_id(uint32_t peer_id);
	peer_snapshot get_all_peers();

	uint32_t insert_message(uint32_t sender_id, uint32_t reader_id, const std::string& content);
	void remove_last_message(uint32_t peer_id, uint32_t self_id);
	std::vector<message> get_messages_for_peer(uint32_t peer_id, uint32_t self_id);

	std::vector<lore_row> get_lorebook_for_peer(uint32_t peer_id);
	void insert_lorebook(uint32_t peer_id, const std::string& keyword, const std::string& content);
	void delete_lorebook_entry(int id);
	void update_lorebook_entry(int id, const std::string& keyword, const std::string& content);

  private:
	peer_snapshot load_peers_locked();
	void invalidate_peers_locked();

	sqlite3* db_ = nullptr;
	std::mutex mutex_;
	peer_snapshot peers_cache_;
};


// namespace ends
} // namespace chatty

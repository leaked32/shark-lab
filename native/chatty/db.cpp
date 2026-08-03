/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/db.cpp
 *
 * License: MIT
 */

#include "db.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>

// ==============================================================================================
// CONFIGURATION
// ==============================================================================================

static chatty::color4 load_color(
	boost::json::object& json, std::string_view name, chatty::color4 fallback)
{
	try {
		auto value = json[name].as_array();

		return {
			static_cast<float>(value[0].get_double()), static_cast<float>(value[1].get_double()),
			static_cast<float>(value[2].get_double()), static_cast<float>(value[3].get_double())};
	} catch (std::exception&) {
		return fallback;
	}
}

chatty::config::config(
	std::string_view path) : path_(path)
{
	try {
		auto content = shark::file::read_json(path_);
		peer_id_ = content["peer_id"].get_int64();
		server_address_ = content["server_address"].get_string();
		server_port_ = content["server_port"].get_int64();

		font_ = content["font"].get_string();
		background_path_ = content["background"].get_string();
		dark_mode_ = content["dark_mode"].get_bool();

		if (content.contains("theme")) {
			auto theme = content["theme"].as_object();

			window_bg_ = load_color(theme, "window_bg", window_bg_);
			frame_bg_ = load_color(theme, "frame_bg", frame_bg_);
			button_ = load_color(theme, "button", button_);
			button_hovered_ = load_color(theme, "button_hovered", button_hovered_);
			button_active_ = load_color(theme, "button_active", button_active_);
			text_ = load_color(theme, "text", text_);
		}

		shark::log::debug("peer_id: {}; server_address: {}; server_port: {}", peer_id_,
						  server_address_, server_port_);

	} catch (std::exception& exc) {
		shark::log::debug("Configuration File cannot load properly: {}", path);
		// Defaults
	}
}

void chatty::config::save() const
{
	return;

	boost::json::object content;
	content["peer_id"] = peer_id_;
	content["server_address"] = server_address_;
	content["server_port"] = server_port_;
	shark::file::dump_json(path_, content);
}

chatty::config::~config()
{
	save();
}

// ==============================================================================================
// SQLITE DATABASE
// ==============================================================================================

constexpr const char* SQL_DB_INIT = R"(
CREATE TABLE IF NOT EXISTS peer (
	id              INTEGER PRIMARY KEY AUTOINCREMENT,
	name            TEXT NOT NULL,
	card            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lorebook (
	id          INTEGER PRIMARY KEY AUTOINCREMENT,
	peer_id     INTEGER NOT NULL,
	keyword     TEXT NOT NULL,
	content     TEXT NOT NULL,

	FOREIGN KEY(peer_id)
		REFERENCES peer(id)
		ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message (
	id              INTEGER PRIMARY KEY AUTOINCREMENT,
	sender_id       INTEGER,
	reader_id       INTEGER,
	content         TEXT NOT NULL,

	created_at      TEXT DEFAULT (datetime('now')),

	FOREIGN KEY (reader_id)
	REFERENCES peer(id)
	ON DELETE CASCADE
	ON UPDATE CASCADE,
	
	FOREIGN KEY (sender_id)
	REFERENCES peer(id)
	ON DELETE CASCADE
	ON UPDATE CASCADE
);
)";

chatty::db::db(
	std::string_view path)
{
	this->db_ = nullptr;
	const std::string path_string(path);
	int rc = sqlite3_open(path_string.c_str(), &db_);
	if (rc) {
		shark::raise("Failed to open sqlite database: {} for {}", path, sqlite3_errmsg(db_));
	}
	init();
}

chatty::db::~db()
{
	std::scoped_lock lock(mutex_);
	sqlite3_close(db_);
}

void chatty::db::init()
{
	sqlite3_exec(db_, "PRAGMA foreign_keys = ON;", nullptr, nullptr, nullptr);

	char* errMsg = nullptr;
	int rc = sqlite3_exec(db_, SQL_DB_INIT, nullptr, nullptr, &errMsg);

	if (rc != SQLITE_OK) {
		shark::raise("Error creating tables: {}", errMsg);
		sqlite3_free(errMsg);
	}
	else {
		shark::log::info("Table craeted. ");
	}
}

void chatty::db::update_peer(
	uint32_t id, const std::string& name, const std::string& card)
{
	std::scoped_lock lock(mutex_);

	const char* sql = R"(
        UPDATE peer SET name = ?, card = ? WHERE id = ?;
    )";

	sqlite3_stmt* stmt = nullptr;

	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
		throw std::runtime_error(sqlite3_errmsg(db_));

	sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt, 2, card.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_int(stmt, 3, id);

	rc = sqlite3_step(stmt);
	sqlite3_finalize(stmt);

	if (rc != SQLITE_DONE) {
		shark::raise("Update peer failed: {}", sqlite3_errmsg(db_));
	}

	invalidate_peers_locked();
}

uint32_t chatty::db::insert_peer(
	const std::string& name, const std::string& card)
{
	std::scoped_lock lock(mutex_);
	const char* sql = R"(
        INSERT INTO peer (name, card) VALUES (?, ?)
    )";

	sqlite3_stmt* stmt = nullptr;

	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare failed: {}", sqlite3_errmsg(db_));
	}

	sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt, 2, card.c_str(), -1, SQLITE_TRANSIENT);

	rc = sqlite3_step(stmt);
	sqlite3_finalize(stmt);

	if (rc != SQLITE_DONE) {
		shark::raise("Insert peer failed: {}", sqlite3_errmsg(db_));
	}

	const auto id = static_cast<uint32_t>(sqlite3_last_insert_rowid(db_));
	invalidate_peers_locked();
	return id;
}

uint32_t chatty::db::insert_message(
	uint32_t sender_id, uint32_t reader_id, const std::string& content)
{
	std::scoped_lock lock(mutex_);

	const char* sql = R"(
		INSERT INTO message (sender_id, reader_id, content) VALUES (?, ?, ?)
	)";

	sqlite3_stmt* stmt = nullptr;
	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare failed: {}", sqlite3_errmsg(db_));
	}

	// Correct binding
	sqlite3_bind_int(stmt, 1, static_cast<int>(sender_id)); // safe cast
	sqlite3_bind_int(stmt, 2, static_cast<int>(reader_id));
	sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_TRANSIENT);

	rc = sqlite3_step(stmt);
	sqlite3_finalize(stmt);

	if (rc != SQLITE_DONE) {
		shark::raise("Insert message failed: {}", sqlite3_errmsg(db_));
	}

	uint32_t last_id = sqlite3_last_insert_rowid(db_);
	return last_id;
}

chatty::db::peer_snapshot chatty::db::load_peers_locked()
{
	if (peers_cache_) {
		return peers_cache_;
	}

	auto peers = std::make_shared<peer_list>();

	const char* sql = "SELECT id, name, card FROM peer ORDER BY id";

	sqlite3_stmt* stmt = nullptr;
	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare get_all_peers failed: {}", sqlite3_errmsg(db_));
	}

	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
		peer p;
		p.id = static_cast<uint32_t>(sqlite3_column_int(stmt, 0));
		const unsigned char* name = sqlite3_column_text(stmt, 1);
		const unsigned char* card = sqlite3_column_text(stmt, 2);
		p.name = name ? reinterpret_cast<const char*>(name) : "";
		p.card = card ? reinterpret_cast<const char*>(card) : "";
		peers->push_back(std::move(p));
	}

	if (rc != SQLITE_DONE) {
		const std::string error = sqlite3_errmsg(db_);
		sqlite3_finalize(stmt);
		shark::raise("Read peers failed: {}", error);
	}

	sqlite3_finalize(stmt);
	peers_cache_ = std::move(peers);
	return peers_cache_;
}

void chatty::db::invalidate_peers_locked()
{
	peers_cache_.reset();
}

chatty::db::peer_snapshot chatty::db::get_all_peers()
{
	std::scoped_lock lock(mutex_);
	return load_peers_locked();
}

std::vector<chatty::message> chatty::db::get_messages_for_peer(
	uint32_t peer_id, uint32_t self_id)
{
	std::scoped_lock lock(mutex_);

	std::vector<message> messages;

	const char* sql = R"(
		SELECT id, sender_id, reader_id, content, created_at
		FROM message
		WHERE (sender_id = ? AND reader_id = ?) OR (reader_id = ? AND sender_id = ?)
		ORDER BY created_at ASC
	)";

	sqlite3_stmt* stmt = nullptr;
	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare get_messages_for_peer failed: {}", sqlite3_errmsg(db_));
	}

	sqlite3_bind_int(stmt, 1, static_cast<int>(peer_id));
	sqlite3_bind_int(stmt, 2, static_cast<int>(self_id));
	sqlite3_bind_int(stmt, 3, static_cast<int>(peer_id));
	sqlite3_bind_int(stmt, 4, static_cast<int>(self_id));

	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
		message m;
		m.id = static_cast<uint32_t>(sqlite3_column_int(stmt, 0));
		m.sender_id = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));
		m.reader_id = static_cast<uint32_t>(sqlite3_column_int(stmt, 2));
		const unsigned char* content = sqlite3_column_text(stmt, 3);
		const unsigned char* time = sqlite3_column_text(stmt, 4);

		m.content = content ? reinterpret_cast<const char*>(content) : "";
		m.created_at = time ? reinterpret_cast<const char*>(time) : "";

		messages.push_back(std::move(m));
	}

	sqlite3_finalize(stmt);
	return messages;
}

void chatty::db::remove_peer(
	uint32_t peer_id)
{
	std::scoped_lock lock(mutex_);

	sqlite3_stmt* stmt = nullptr;
	// delete peer
	const char* sql = "DELETE FROM peer WHERE id = ?;";
	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare failed: {}", sqlite3_errmsg(db_));
	}
	sqlite3_bind_int(stmt, 1, peer_id);

	rc = sqlite3_step(stmt);
	sqlite3_finalize(stmt);

	if (rc != SQLITE_DONE) {
		shark::raise("Remove peer failed: {}", sqlite3_errmsg(db_));
	}

	invalidate_peers_locked();
}

void chatty::db::remove_last_message(
	uint32_t peer_id, uint32_t self_id)
{
	std::scoped_lock lock(mutex_);

	sqlite3_stmt* stmt = nullptr;

	const char* sql =
		R"(
		DELETE FROM message
		WHERE id = (
			SELECT id
			FROM message
			WHERE (sender_id = ? AND reader_id = ?) OR (sender_id = ? AND reader_id = ?)
			ORDER BY id DESC
			LIMIT 1
		);
		)";

	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		shark::raise("Prepare failed: {}", sqlite3_errmsg(db_));
	}

	sqlite3_bind_int(stmt, 1, self_id);
	sqlite3_bind_int(stmt, 2, peer_id);
	sqlite3_bind_int(stmt, 3, peer_id);
	sqlite3_bind_int(stmt, 4, self_id);

	sqlite3_step(stmt);
	sqlite3_finalize(stmt);
}

std::optional<chatty::peer> chatty::db::get_peer_by_id(
	uint32_t peer_id)
{
	const auto peers = get_all_peers();
	const auto it = std::ranges::find(*peers, peer_id, &peer::id);
	return it == peers->end() ? std::nullopt : std::optional<peer>{*it};
}

std::vector<chatty::lore_row> chatty::db::get_lorebook_for_peer(
	uint32_t peer_id)
{
	std::scoped_lock lock(mutex_);

	const char* sql = "SELECT id, keyword, content FROM lorebook WHERE peer_id "
					  "= ? ORDER BY id ASC;";
	sqlite3_stmt* stmt = nullptr;
	if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
		throw std::runtime_error("Failed to prepare get_lorebook statement");

	sqlite3_bind_int(stmt, 1, peer_id);

	std::vector<lore_row> result;
	while (sqlite3_step(stmt) == SQLITE_ROW) {
		lore_row row;
		row.id = sqlite3_column_int(stmt, 0);
		row.keyword = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
		row.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
		result.push_back(row);
	}

	sqlite3_finalize(stmt);
	return result;
}

void chatty::db::insert_lorebook(
	uint32_t peer_id, const std::string& keyword, const std::string& content)
{
	std::scoped_lock lock(mutex_);

	const char* sql = "INSERT INTO lorebook (peer_id, keyword, content) VALUES (?, ?, ?);";
	sqlite3_stmt* stmt = nullptr;
	if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
		throw std::runtime_error("Failed to prepare insert_lorebook statement");

	sqlite3_bind_int(stmt, 1, peer_id);
	sqlite3_bind_text(stmt, 2, keyword.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_TRANSIENT);

	if (sqlite3_step(stmt) != SQLITE_DONE) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Failed to insert lorebook entry");
	}

	sqlite3_finalize(stmt);
}

void chatty::db::delete_lorebook_entry(
	int id)
{
	std::scoped_lock lock(mutex_);

	const char* sql = "DELETE FROM lorebook WHERE id = ?;";
	sqlite3_stmt* stmt = nullptr;
	if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
		throw std::runtime_error("Failed to prepare delete_lorebook_entry statement");

	sqlite3_bind_int(stmt, 1, id);

	if (sqlite3_step(stmt) != SQLITE_DONE) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Failed to delete lorebook entry");
	}

	sqlite3_finalize(stmt);
}

void chatty::db::update_lorebook_entry(
	int id, const std::string& keyword, const std::string& content)
{
	std::scoped_lock lock(mutex_);

	const char* sql = R"(
        UPDATE lorebook
        SET keyword = ?,
            content = ?
        WHERE id = ?;
    )";

	sqlite3_stmt* stmt = nullptr;

	int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		throw std::runtime_error(sqlite3_errmsg(db_));
	}

	sqlite3_bind_text(stmt, 1, keyword.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt, 2, content.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_int(stmt, 3, id);

	rc = sqlite3_step(stmt);
	if (rc != SQLITE_DONE) {
		sqlite3_finalize(stmt);
		throw std::runtime_error(sqlite3_errmsg(db_));
	}

	sqlite3_finalize(stmt);
}

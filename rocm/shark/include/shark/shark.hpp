/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: shark/include/shark/shark.h
 *
 * License: MIT
 */

#pragma once

#ifndef __cplusplus
#error "This header requires C++"
#endif

#include "shark/common.hpp"

#include <list>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <map>
#include <coroutine>
#include <exception>
#include <algorithm>
#include <stack>
#include <iostream>
#include <format>
#include <random>

#include <boost/json.hpp>

SHARK_BEGIN

constexpr bool LIT_DEBUG = true;

namespace log
{
	// inline std::mutex _logger_bus;

	// Your message_type enum stays compatible
	enum class message_type {
		INFO,
		DEBUG,
		EXCEPTION
	};
	

	// Internal backend class
	class backend {
	public:
		backend() : worker_([this]{ run(); }) {}
	
		~backend() {
			shutdown(); // automatic flush and stop
		}
	
		backend(const backend&) = delete;
		backend& operator=(const backend&) = delete;
	
		// Push a message into the queue
		void push(message_type type, std::string msg) {
			{
				std::lock_guard lock(mtx_);
				queue_.emplace(type, std::move(msg));
			}
			cv_.notify_one();
		}
	
		// Explicit shutdown (optional)
		void shutdown() {
			bool expected = false;
			if (!stopped_.compare_exchange_strong(expected, true))
				return; // already stopped
	
			{
				std::lock_guard lock(mtx_);
				stopping_ = true;
			}
	
			cv_.notify_all();
	
			if (worker_.joinable())
				worker_.join();
		}
	
	private:
		void run() {
			for (;;) {
				std::unique_lock lock(mtx_);
				cv_.wait(lock, [&]{ return stopping_ || !queue_.empty(); });
	
				if (stopping_ && queue_.empty())
					break;
	
				auto [type, msg] = std::move(queue_.front());
				queue_.pop();
	
				lock.unlock();
	
				// Output exactly the message string
				std::cout << msg << std::endl;
			}
		}
	
	private:
		std::mutex mtx_;
		std::condition_variable cv_;
		std::queue<std::pair<message_type, std::string>> queue_;
		bool stopping_ = false;
		std::atomic_bool stopped_ = false;
		std::thread worker_;
	};
	
	// Function-local static instance ensures automatic creation
	inline backend& instance()
	{
		static backend b;
		return b;
	}
	
	// Submit API matches your previous API exactly
	inline void submit(message_type type, const std::string& msg)
	{
		instance().push(type, msg);
	}
	
	inline void submit(message_type type, std::string&& msg)
	{
		instance().push(type, std::move(msg));
	}
	
	// Optional kill() function for explicit shutdown
	inline void kill()
	{
		instance().shutdown();
	}
	

	template <typename... Types>
	constexpr void info(const std::format_string<Types...> &Fmt, Types &&...Args)
	{
		std::string info_str = std::format(Fmt, std::forward<Types>(Args)...);
		submit(message_type::INFO, info_str);

		// std::string info_str = std::format("INFO:  {}", std::format(Fmt, std::forward<Types>(Args)...));
		// submit(info_str);
	}


	template <typename... Types>
	constexpr void debug(const std::format_string<Types...> &Fmt, Types &&...Args)
	{
		// print<"INFO: {}">(std::format(Fmt, std::forward<Types>(Args)...));
		if constexpr (LIT_DEBUG)
		{
			std::string info_str = std::format(Fmt, std::forward<Types>(Args)...);
			submit(message_type::DEBUG, info_str);

		}
	}

	template <typename... Types>
	constexpr void exception(const std::format_string<Types...> &Fmt, Types &&...Args)
	{
		std::string info_str = std::format(Fmt, std::forward<Types>(Args)...);
		submit(message_type::EXCEPTION, info_str);
	}
}


template <typename... Types>
constexpr void print(const std::format_string<Types...> &Fmt, Types &&...Args)
{
	std::string info_str = std::format(Fmt, std::forward<Types>(Args)...);
	log::submit(log::message_type::INFO, info_str);
}

void indented_print(std::string_view s, char c = ' ', int n = 4);
std::string indent(std::string_view s, char c = ' ', int n = 4);


struct pinch : std::exception
{
	template <typename... Types>
	constexpr pinch(const std::format_string<Types...> &Fmt, Types &&...Args)
	{
		std::string info_str = std::format(Fmt, std::forward<Types>(Args)...);
		msg_ = std::format("trace {}:{}", SHARK_FUNCTION, info_str);
	}
	// pinch(std::string_view func_name, std::string_view information) : std::exception{}, msg_(std::format("trace {}:{}", func_name, information))
	// { }
	
	pinch() : std::exception{}, msg_("{null}") {}
	
	virtual const char *what() const noexcept override
	{
		return msg_.c_str();
	}
	
	std::string msg_;
};

template <typename... Types>
void raise(const std::format_string<Types...> &Fmt, Types &&...Args)
{
	shark::print(Fmt, std::forward<Types>(Args)...);
	throw pinch(Fmt, std::forward<Types>(Args)...);
}

#undef assert
template <typename... Types>
void assert(bool condition, const std::format_string<Types...> &Fmt, Types &&...Args)
{
	if (!condition) {
		raise(Fmt, std::forward<Types>(Args)...);
	}
}

inline void assert(bool condition)
{
	assert(condition, "");
}



// For specified error code
template <auto F, auto N, typename... Types>
auto checked_invoke(Types &&...Args)
{
	auto result = F(std::forward<Types>(Args)...);
	if (decltype(N)(result) != N)
	{
		log::debug("Invocation of function has failed: {} {:#x}", SHARK_FUNCTION, decltype(N)(result));
		throw pinch{"Invocation of function has failed"};
	}
	return result;
}

template <auto F, typename... Types>
auto yinvoke(Types &&...Args)
{
	return checked_invoke<F, 0UL>(std::forward<Types>(Args)...);
}

template <auto F, typename... Types>
auto einvoke(Types &&...Args)
{
	return checked_invoke<F, 0UL>(std::forward<Types>(Args)...);
}

template<typename T, T good>
struct rcheck
{
	void check(T v) const {
		if (v != good) {
			log::debug("Invocation of function has failed: {} {:#x}",
					   SHARK_FUNCTION, static_cast<unsigned long>(v));
			throw pinch{"Invocation of function has failed"};
		}
	}
	template<typename... Types>
	void operator=(Types&& ...values) const {
		check(std::forward<Types>(values)...);
	}
};


struct profiler
{
	using clock = std::chrono::steady_clock;
	using time_point = clock::time_point;
	using duration = std::chrono::duration<double>;
	
	profiler(size_t reserve, bool print_on_lap) :
			start(clock::now()), print_on_lap(print_on_lap) {
		laps.reserve(reserve);
	}
	
	void lap(std::optional<std::string_view> msg = std::nullopt)
	{
		time_point end = clock::now();
		const duration diff = end - start;
		laps.emplace_back(diff);
		
		if (print_on_lap) {
			print_duration(diff, msg);
		}
		
		reset(); // exclude emplace_back
	}
	
	void reset() {
		start = clock::now();
	}
	
	static void print_duration(duration value, std::optional<std::string_view> msg = std::nullopt)
	{
		const auto milliseconds =
		std::chrono::duration<double, std::milli>(value).count();
		
		if (msg.has_value()) {
			log::info("{}: {:.6f} ms", msg.value(), milliseconds);
		} else {
			log::info("{:.6f} ms", milliseconds);
		}
	}
	
	time_point start;
	bool print_on_lap;
	std::vector<duration> laps;
};

void forcely_print_string(const std::string &input);
void forcely_print_vector(const std::vector<std::string> &input);


namespace file
{
	class tlm_node {
		// Touhou Little Marks
	public:
		void set_doc(std::string_view content) {
			doc = std::make_unique<std::string>(content);
		}
		
		bool has_doc() const {
			return doc != nullptr;
		}
		
		std::optional<std::string_view> get_doc() const {
			return doc != nullptr ?
				std::optional<std::string_view>{*doc} : std::optional<std::string_view>{};
		}
		
		// std::string_view doesn't gaurantee null-terminate
		tlm_node& get_child(std::string_view key) {
			if (children == nullptr) {
				children = std::make_unique<decltype(children)::element_type>();
			}
			
			
			auto p = children->try_emplace(std::string(key));
			return p.first->second;
		}
		
		template<typename Ty>
		tlm_node& operator[] (Ty&& key) {
			return get_child(std::forward<Ty>(key));
		}
		
		std::string print(unsigned short indent = 0) const;
		
		std::optional<std::reference_wrapper<const std::map<std::string, tlm_node>>> c_get_children() const {
			if (children) {
				return *children;
			}
			return {};
		}
	private:
		
		// lit::print("{}  {}", sizeof (std::map<std::string, tlm_node>), sizeof (std::unique_ptr<std::map<std::string, tlm_node>>));
		std::unique_ptr<std::map<std::string, tlm_node>> children = nullptr;
		std::unique_ptr<std::string> doc = nullptr;
	};
	
	std::string read(std::string_view path);
	std::vector<char> read_binary(const std::string& filename);
	tlm_node tlm_node_read(std::string_view path);
	boost::json::object read_json(std::string_view path);
	void dump_json(std::string_view path, const boost::json::object& js);
	
};

namespace str
{
	std::string replace(std::string_view input, std::string_view from, std::string_view to);
	std::vector<std::string> split(std::string_view input, std::string_view delimiter);
	std::optional<std::string> trim(const std::string& s);
	unsigned short leading_space_count(const std::string& s);
	bool is_empty_or_whitespace(const std::string& s);
	void remove_whitespace(std::string& s);
	size_t utf8_codepoints(std::string_view s);
	size_t estimate_tokens(std::string_view s);
	std::string to_lower(const std::string &s);
}


namespace math
{
	
	namespace
	{
		struct _internal
		{
			std::random_device rd;
			std::mt19937 gen;
			std::uniform_real_distribution<float> dis;
			
			_internal() : gen(rd()) {}
		};
		
		static std::unique_ptr<_internal> static_random_ = nullptr;
		static _internal &instance()
		{
			if (!static_random_)
			{
				static_random_ = std::make_unique<_internal>();
			}
			return *static_random_;
		}
	}
	
	float random_float(const float begin = 0.0f, const float end = 1.0f);
	int random_int(const int begin, const int end);
	
	template <typename T>
	std::tuple<T, T> from_interval_text(std::string_view interval)
	{
		std::string intr = str::replace(interval, " ", "");
	}
}

namespace async
{
	struct io_context;
	
	struct delayer
	{
		std::chrono::system_clock::duration timer_;
		void reset()
		{
			timer_ = std::chrono::seconds(2);
		}
		bool before(std::chrono::system_clock::time_point end_point)
		{
			auto at_least = end_point - std::chrono::system_clock::now();
			// std::chrono::system_clock::duration at_least =
			//    end_point - std::chrono::system_clock::now();
			// log::debug("before: at_least {}", at_least);
			if (at_least <= std::chrono::milliseconds(0)) {
				timer_ = std::chrono::milliseconds(0);
				return true;
			}
			else if (at_least < timer_) {
				timer_ = at_least;
				return false;
			}
			return false;
		}

		void exec_delay() const
		{
			if (timer_ == std::chrono::milliseconds(0))
			{
				// log::debug("local_delayer::exec_delay time's up");
				return;
			}
			// log::debug("local_delayer::exec_delay delay");
			std::this_thread::sleep_for(timer_);
		}
		void shorter(std::chrono::system_clock::duration dura)
		{
		}
	};
	
	// task_context, thread slice
	class strand
	{
	public:
		io_context &parent_;
		std::stack<std::coroutine_handle<>> addresses_;
		
		struct Condition
		{
			virtual bool evaluate(delayer &local_delayer) = 0;
			// Original virtual destructor is not allowed to be 0
			virtual ~Condition() = default;
		};
		
		std::atomic_bool proceeding_ = false;
		
		strand(io_context &parent, std::coroutine_handle<> coro) : parent_(parent)
		{
			addresses_.push(coro);
		}
		
		strand(io_context &parent, strand &&obj) noexcept : 
				parent_(parent), addresses_(std::move(obj.addresses_)),
				condition_(std::move(obj.condition_))
		{
			proceeding_ = static_cast<bool>(obj.proceeding_);
		}
		
		auto stacks() const -> std::string
		{
			std::ostringstream ss;
			ss << "[";
			
			auto copy = addresses_;
			std::vector<decltype(copy.top().address())> addresses;
			
			addresses.reserve(copy.size());
			
			while (!copy.empty()) {
				addresses.push_back(copy.top().address());
				copy.pop();
			}
			
			for (auto it = addresses.rbegin(); it != addresses.rend(); ++it) {
				if (it != addresses.rbegin()) {
					ss << ", ";
				}
				ss << *it;
			}
			
			ss << "]";
			return ss.str();
		}
		
		bool evaluate_condition(delayer& d)
		{
			std::scoped_lock lock(condition_mutex_);
			if (!condition_) {
				return true;
			}
			return condition_->evaluate(d);
		}
		
		void set_condition(std::unique_ptr<Condition>&& condition)
		{
			std::scoped_lock lock(condition_mutex_);
			condition_ = std::move(condition);
		}
	private:
		std::unique_ptr<Condition> condition_ = nullptr;
		std::mutex condition_mutex_;
	};
	
	struct promise_type_base
	{
		std::weak_ptr<strand> slice_ ;
		std::exception_ptr except_ = nullptr;
	};
	
	template <typename Result>
	struct awaitable
	{
		struct promise_type;
		
		using Handle_Type = std::coroutine_handle<promise_type>;
		Handle_Type coro_handle_;
		
		struct promise_type : promise_type_base
		{
			std::unique_ptr<Result> result_ = nullptr;
			
			auto get_return_object() { return awaitable{Handle_Type::from_promise(*this)}; }
			
			// I prefer not to mark them static
			constexpr std::suspend_always initial_suspend() { return {}; }
			constexpr std::suspend_always final_suspend() noexcept { return {}; }
			
			template <typename... Types>
			void return_value(Types &&...args)
			{
				result_ = std::make_unique<Result>(std::forward<Types>(args)...);
			}
			
			void unhandled_exception()
			{
				// I shall forward the exception, and terminate the thread.
				try
				{
					except_ = std::current_exception();
					std::rethrow_exception(except_);
				}
				catch (std::exception &exc)
				{
					auto slice = slice_.lock();
					
					if (slice) {
						log::debug(
							"slice: {}; current coroutine: {}; exception what: {}",
							(unsigned long)slice.get(), 
							(unsigned long)slice->addresses_.top().address(), exc.what()
						);
					}
				}
				std::terminate();
			}
		};
		
		void resume()
		{
			coro_handle_.resume();
		}
		~awaitable()
		{
			// if(coro_handle_) {
			// 	coro_handle_.destroy();
			// 	coro_handle_ = nullptr;
			// }
		}
		
#pragma region Awaitable_Implementation
		
		constexpr auto await_ready() { return false; }
		void await_suspend(std::coroutine_handle<> parent_handle)
		{
			// using casted_type = std::coroutine_handle<promise_type_base>;
			// auto *p_parent_handle_base = reinterpret_cast<casted_type *>(&parent_handle);
			auto coro_handle =
				std::coroutine_handle<promise_type_base> ::from_address(parent_handle.address());
			
			// auto slice = coro_handle.promise().slice_;
			std::weak_ptr<strand> former_slice = coro_handle.promise().slice_;
			coro_handle_.promise().slice_ = former_slice;
			former_slice.lock()->addresses_.push(coro_handle_);
			
			// h_.resume behaves as a shortcut, it's not required for normal running,
			// since the stack has been pushed. But it will be much faster if I call it here.
			coro_handle_.resume();
		}
		Result await_resume()
		{
			Result result = std::move(*coro_handle_.promise().result_);
			coro_handle_.destroy();
			coro_handle_ = nullptr;
			return result;
		}
		

#pragma endregion
	};
	
	struct io_context
	{
		std::vector<std::shared_ptr<strand>> context_;
		// std::list<thread_slice> context_;
		std::mutex mtx_context_;
		
		std::shared_ptr<strand> get_ready_coroutine()
		{
			std::scoped_lock mtx_context_lock(mtx_context_);
			
			using context_it = std::vector<std::shared_ptr<strand>>::iterator;
			
			for (context_it i = context_.begin(); i != context_.end(); ++i) {
				// exchange will return the previous value,
				// if it's true, the coroutine is already running.
				if ((*i)->proceeding_.exchange(true)) {
					continue;
				}
				return *i;
			}
			return nullptr;
		}
		
		void erase_coroutine(const std::shared_ptr<strand>& target)
		{
			std::scoped_lock lock(mtx_context_);
			
			std::erase_if(
				context_,
				[&](auto& x) {
					return x == target;
				}
			);
		}
		
		template <typename TAsyncAwaitable>
		void spawn(TAsyncAwaitable &&coro)
		{
			std::scoped_lock mtx_context_lock(mtx_context_);
			
			std::shared_ptr<strand> slice = std::make_shared<strand>(*this, coro.coro_handle_);
			coro.coro_handle_.promise().slice_ = slice;
			context_.emplace_back(std::move(slice));
		}
		
		bool empty()
		{
			std::scoped_lock lock(mtx_context_);
			return context_.empty();
		}
		
		void run()
		{
			delayer local_delayer{};
			while (!empty()) {
				local_delayer.reset();
				// log::debug("local_delayer reset");
				
				std::shared_ptr<strand> sit = get_ready_coroutine();
				// Usually, if a thread is here, there should be something to execute.
				// Because the delayer automatically delays for the span to the the most
				// earliest available coroutines.
				if (!sit) {
					// local_delayer.exec_delay();
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					std::this_thread::yield();
					continue;
				}
				
				// log::debug("selected {}", sit->stacks());
				// log::debug("mtx_context_lock unlock");
				// If the condition is true, then the thread hasn't bound to
				// any specific thread_slice currently.
				
				while (true) {
					if (!sit->evaluate_condition(local_delayer)) {
						break;
					}
					// Coroutine can call others and emplace them to the stack in io_context,
					// but it mustn't resume coroutines behind.
					// log::debug("resuming {}", sit->stacks());
					sit->addresses_.top().resume();
					
					// The most efficient way for it:
					// direct execute the coroutine by co_await, 
					// and append it into the context list.
					
					// Case 1: The top coroutine appears to be done
					// The struct ensures that only the top coroutine can be done at one execution.
					if (auto &current_top = sit->addresses_.top(); current_top.done()) {
						// destroy() should be called after value is obtained, not here.
						sit->addresses_.pop();
						if (sit->addresses_.empty()) {
							// There's no other parents could destroy the root coroutine.
							current_top.destroy();
							erase_coroutine(sit);
							break;
						}
					}
					// Case 2: The top coroutine hasn't completed, 
					// but still returned handling back.
					// I shall go back to check the conditions...
					// If the condition is still none, 
					// there's possibility that the coroutine just invoked a new one, 
					// but hierarchy restrict from execution of parent coroutine.
				
				}
				// sit->proceeding_ must be set after while(true), since it keeps executing it.
				// otherwise other threads may obtain it sneakily.
				sit->proceeding_ = false;
				local_delayer.exec_delay();
			}
			log::debug(
				"io_context::run has ended since no candidate async action is provided\n",
				"I don't think it's something you expect"
			);
		}
	};
	
	struct deadline_timer
	{
		// Little struct does not necessarily need move constructor
		struct deadline_timer_condition : strand::Condition
		{
			std::chrono::system_clock::time_point end_time_point_;
			bool evaluate(delayer &local_delayer) override
			{
				return local_delayer.before(end_time_point_);
			}
			~deadline_timer_condition() override = default;
			deadline_timer_condition(std::chrono::system_clock::time_point end_time_point) :
				end_time_point_(end_time_point) {}
		};
		
		struct deadline_timer_awaitable
		{
			deadline_timer_condition condition_;
			constexpr bool await_ready() { return false; }
			void await_suspend(std::coroutine_handle<> parent_handle)
			{
				auto parent_coro_handle =
				std::coroutine_handle<promise_type_base>
				::from_address(parent_handle.address());
				
				parent_coro_handle.promise().slice_.lock()->set_condition(
					std::make_unique<deadline_timer_condition>(condition_)
				);
			}
			constexpr void await_resume() {}
		};
		
		auto expires_after(std::chrono::system_clock::duration dura)
		{
			return deadline_timer_awaitable{{std::chrono::system_clock::now() + dura}};
			// NOTE: use steady_clock for better performance
		}
	};
	
	template<typename Job>
	class multi_executor
	{
	public:
		using job_type = std::function<Job>;
		
		multi_executor(size_t thread_count)
		{
			for (size_t i = 0; i < thread_count; ++i) {
				ioc.spawn(async_load(i));
			}
			
			for (size_t i = 0; i < thread_count; ++i) {
				threads.emplace_back(
					[this]
					{
						ioc.run();
					}
				);
			}
		}
		
		void push(job_type job)
		{
			std::scoped_lock lock(queue_mutex_);
			queue_.push(std::move(job));
		}
		
		void shutdown()
		{
			shutting_down_ = true;
		}
		
	private:
		
		shark::async::awaitable<int> async_load(int id)
		{
			// log::debug("Executing the job 0");
			shark::async::deadline_timer timer;
			
			while (!shutting_down_) {
				job_type job;
				
				{
					std::scoped_lock lock(queue_mutex_);
					if (!queue_.empty())
					{
						job = std::move(queue_.front());
						queue_.pop();
					}
				}
				// log::debug("Executing the job 1");
				if (job) {
					// log::debug("Executing the job 2");
					job();
				}
				co_await timer.expires_after(
					std::chrono::milliseconds(100)
				);
			}
			// log::debug("Executing the job 2");
			co_return 0;
		}
		
		std::queue<job_type> queue_;
		std::mutex queue_mutex_;
		
		std::atomic_bool shutting_down_ = false;
		
		shark::async::io_context ioc;
		std::vector<std::thread> threads;
	};

}


SHARK_END

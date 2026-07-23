
#include "shark/common.h"

#include <hip/amd_detail/amd_hip_vector_types.h>
#include <random>
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
	void check(T v) {
		if (v != good) {
			log::debug("Invocation of function has failed: {} {:#x}",
					   SHARK_FUNCTION, static_cast<unsigned long>(v));
			throw pinch{"Invocation of function has failed"};
		}
	}
	template<typename... Types>
	void operator=(Types&& ...values) {
		check(std::forward<Types>(values)...);
	}
};


void forcely_print_string(const std::string &input);
void forcely_print_vector(const std::vector<std::string> &input);


class file
{
public:
	
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
	
	static std::string read(std::string_view path);


	static std::vector<char> read_binary(const std::string& filename);
	
	static tlm_node tlm_node_read(std::string_view path);

	static boost::json::object read_json(std::string_view path);
	static void dump_json(std::string_view path, const boost::json::object& js);


private:
	
};

std::string str_replace(std::string_view input, std::string_view from, std::string_view to);

std::vector<std::string> str_split(std::string_view input, std::string_view delimiter);

std::optional<std::string> str_trim(const std::string& s);

unsigned short leading_space_count(const std::string& s);

bool is_empty_or_whitespace(const std::string& s);

void remove_whitespace(std::string& s);

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
		std::string intr = str_replace(interval, " ", "");
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
			// std::chrono::system_clock::duration at_least = end_point - std::chrono::system_clock::now();
			// log::debug("before: at_least {}", at_least);
			if (at_least <= std::chrono::milliseconds(0))
			{
				timer_ = std::chrono::milliseconds(0);
				return true;
			}
			else if (at_least < timer_)
			{
				timer_ = at_least;
				return false;
			}
			return false;
		}

		void exec_delay() const
		{
			if (timer_ == std::chrono::milliseconds(0))
			{
				log::debug("local_delayer::exec_delay time's up");
				return;
			}
			log::debug("local_delayer::exec_delay delay");
			std::this_thread::sleep_for(timer_);
		}
		void shorter(std::chrono::system_clock::duration dura)
		{
		}
	};

	struct thread_slice
	{
		io_context &parent_;
		std::stack<std::coroutine_handle<>> addresses_;
		
		struct Condition
		{
			virtual bool evaluate(delayer &local_delayer) = 0;
			// Original virtual destructor is not allowed to be 0
			virtual ~Condition() = default;
		};
		
		std::unique_ptr<Condition> condition_ = nullptr;
		std::atomic_bool proceeding_ = false;
		
		thread_slice(io_context &parent, std::coroutine_handle<> coro) : parent_(parent)
		{
			addresses_.push(coro);
		}
		
		thread_slice(io_context &parent, thread_slice &&obj) noexcept : parent_(parent), addresses_(std::move(obj.addresses_)),
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
			
			while (!copy.empty())
			{
				addresses.push_back(copy.top().address());
				copy.pop();
			}
			
			for (auto it = addresses.rbegin(); it != addresses.rend(); ++it)
			{
				if (it != addresses.rbegin())
				{
					ss << ", ";
				}
				
				ss << *it;
			}
			
			ss << "]";
			return ss.str();
		}
	};
	
	struct promise_type_base
	{
		thread_slice *slice_ = nullptr;
		std::exception_ptr except_ = nullptr;
	};
	
	template <typename Result>
	struct awaitable
	{
		struct promise_type;
		
		using Handle_Type = std::coroutine_handle<promise_type>;
		Handle_Type h_;
		
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
				// Unhandled exception will definitely have chance to cause something not clearable.
				// I don't assume someone deliberately programs like that, so I choose to dump the error and terminate the thread.
				// I shall forward the exception, and terminate the thread.
				try
				{
					except_ = std::current_exception();
					std::rethrow_exception(except_);
				}
				catch (std::exception &exc)
				{
					log::debug("slice: {}; current coroutine: {}; exception what: {}", (unsigned long)slice_, (unsigned long)slice_->addresses_.top().address(), exc.what());
				}
				std::terminate();
			}
		};
		
		void resume()
		{
			h_.resume();
		}
		
#pragma region Awaitable_Implementation
		
		constexpr auto await_ready() { return false; }
		void await_suspend(std::coroutine_handle<> parent_handle)
		{
			using casted_type = std::coroutine_handle<promise_type_base>;
			auto *p_parent_handle_base = reinterpret_cast<casted_type *>(&parent_handle);
			thread_slice *former_slice = p_parent_handle_base->promise().slice_;
			h_.promise().slice_ = former_slice;
			former_slice->addresses_.push(h_);
			
			// h_.resume behaves as a shortcut, it's not required for normal running, since the stack has been pushed. But it will be much faster if I call it here.
			h_.resume();
		}
		Result await_resume()
		{
			Result result = std::move(*h_.promise().result_);
			h_.destroy();
			return result;
		}

#pragma endregion
	};
	
	struct io_context
	{
		std::list<thread_slice> context_;
		std::mutex mtx_context_;
		
		void run()
		{
			delayer local_delayer{};
			while (context_.empty() == false)
			{
				local_delayer.reset();
				// log::info("local_delayer reset");
				
				std::unique_lock mtx_context_lock(mtx_context_);
				for (auto it = context_.begin(); it != context_.end(); ++it, mtx_context_lock.lock())
				{
					
					if (it->proceeding_ == true)
					{
						continue;
					}
					mtx_context_lock.unlock();
					
					it->proceeding_ = true;
					// log::info("selected {}", it->stacks());
					// log::info("mtx_context_lock unlock");
					auto sit = it;
					// If the condition is true, then the thread hasn't bound to any specific thread_slice currently.
					
					while (true)
					{
						if (sit->condition_ && !sit->condition_->evaluate(local_delayer))
						{
							break;
						}
						// Coroutine can call others and emplace them to the stack in io_context, but it mustn't resume coroutines behind.
						log::debug("resuming {}", sit->stacks());
						sit->addresses_.top().resume();
						
						// The most efficient way for it: direct execute the coroutine by co_await, and append it into the context list.
						
						// Case 1: The top coroutine appears to be done
						// The struct ensures that only the top coroutine can be done at one execution.
						if (auto &current_top = sit->addresses_.top(); current_top.done())
						{
							// sit->addresses_.top().destroy(); // ? Can I call this ? Definitely NO. If I do this, how can the previous coroutine access the result?
							sit->addresses_.pop();
							if (sit->addresses_.empty())
							{
								// There's no other parents could destroy the root coroutine.
								current_top.destroy();
								it--;
								context_.erase(sit);
								break;
							}
						}
						// Case 2: The top coroutine hasn't completed, but still returned handling back.
						// I shall go back to check the conditions...
						// If the condition is still none, there's possibility that the coroutine just invoked a new one, but hierarchy restrict from execution of parent coroutine.
					}
					sit->proceeding_ = false;
				}
				
				local_delayer.exec_delay();
				// std::this_thread::sleep_for(std::chrono::seconds(1));
			}
			log::debug("io_context::run has ended since no candidate async action is provided. I don't think it's something you expected. ");
		}
		
		template <typename TAsyncAwaitable>
		void spawn(TAsyncAwaitable &&coro)
		{
			// coro.h_.promise().slice_ = this;
			auto &slice = context_.emplace_back(*this, coro.h_);
			coro.h_.promise().slice_ = &slice;
		}
	};
	
	struct deadline_timer
	{
		// Little struct does not necessarily need move convention
		struct deadline_timer_condition : thread_slice::Condition
		{
			std::chrono::system_clock::time_point end_time_point_;
			bool evaluate(delayer &local_delayer) override
			{
				return local_delayer.before(end_time_point_);
			}
			~deadline_timer_condition() override = default;
			deadline_timer_condition(std::chrono::system_clock::time_point end_time_point) : end_time_point_(end_time_point) {}
		};
		
		struct deadline_timer_awaitable
		{
			deadline_timer_condition condition_;
			constexpr bool await_ready() { return false; }
			void await_suspend(std::coroutine_handle<> parent_handle)
			{
				using casted_type = std::coroutine_handle<promise_type_base>;
				auto *p_parent_handle_base = reinterpret_cast<casted_type *>(&parent_handle);
				p_parent_handle_base->promise().slice_->condition_ = std::make_unique<deadline_timer_condition>(condition_);
			}
			constexpr void await_resume() {}
		};
		
		auto expires_after(std::chrono::system_clock::duration dura)
		{
			return deadline_timer_awaitable{{std::chrono::system_clock::now() + dura}};
			// I don't think it's a good idea. I thought out another way to implement it, I shall maintain an object that decides when to resume execution for each thread, it should be thread local only, since something is considered to be "event trigger? "
			// Wait, constantly looping is definitely not I want, I shall still implement the system_clock based delay, and I should make it suspended if its event is not depleted.
		}
	};

}


SHARK_END

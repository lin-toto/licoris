#ifndef EMBEDLIC_PIPELINE_H
#define EMBEDLIC_PIPELINE_H

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <utility>
#include <vector>
#include <memory>
#include <thread>
#include <functional>
#include <future>
#include <array>
#include <unordered_map>

struct ITask {
    virtual ~ITask() = default;
};

struct IdentifiedTask : public virtual ITask {
    int taskId;
    explicit IdentifiedTask(int taskId) : taskId(taskId) {}
};

struct AggregatorAwareTask : public virtual ITask, IdentifiedTask {
    int aggregationId;
    AggregatorAwareTask(int taskId, int aggregationId) : IdentifiedTask(taskId), aggregationId(aggregationId) {}
};

template<int num>
struct AggregatedTask : public ITask {
    int completedCount = 0;
    std::array<std::shared_ptr<AggregatorAwareTask>, num> tasks{};
};

class Pipeline {
public:
    Pipeline(const Pipeline&) = delete;

    virtual void feedTask(std::shared_ptr<ITask> task) = 0;
    virtual bool tryFeedTask(std::shared_ptr<ITask> task) = 0;
    virtual void execute(unsigned int threadId) = 0;

    inline const std::shared_ptr<Pipeline>& next(const std::shared_ptr<Pipeline>& next) {
        // Support simple chain syntax.
        nextPipeline = next;
        return next;
    }

    void executeAsync(unsigned int numThreads) {
        static std::vector<std::future<void>> pendingFutures;
        for (unsigned int i = 0; i < numThreads; i++) {
            pendingFutures.push_back(std::async(std::launch::async, [this, i]() { execute(i); }));
        }
    }

    inline void terminate() { executing = false; }
protected:
    bool executing = true;
    explicit Pipeline(std::shared_ptr<Pipeline> nextPipeline = nullptr) : nextPipeline(std::move(nextPipeline)) {}
    std::shared_ptr<Pipeline> nextPipeline;
};

template<typename AcceptedTask, typename ProducedTask = std::nullptr_t,
        typename = std::enable_if_t<std::conjunction_v<
                std::is_base_of<ITask, AcceptedTask>,
                std::disjunction<std::is_base_of<ITask, ProducedTask>, std::is_same<std::nullptr_t, ProducedTask>>>>>
class PipelineImpl : public Pipeline {
public:
    explicit PipelineImpl(unsigned int maxTasks, std::shared_ptr<Pipeline> nextPipeline = nullptr)
        : maxTasks(maxTasks), Pipeline(std::move(nextPipeline)) {}
    PipelineImpl(const PipelineImpl&) = delete;

    void execute(unsigned int threadId) override {
        while (executing) {
            std::unique_lock<std::mutex> lock(queueLock);
            if (queue.empty()) {
                consumerCv.wait(lock, [this]() { return !this->queue.empty(); });
            }

            auto task = std::move(queue.front());

            queue.pop();
            lock.unlock();
            producerCv.notify_all();

            try {
                auto tasks = work(std::move(task), threadId);

                if constexpr (std::is_same_v<std::nullptr_t, ProducedTask>) {
                    if (!tasks.empty() || this->nextPipeline) {
                        throw std::runtime_error("A terminating pipeline cannot have a next or produce tasks");
                    }
                } else {
                    if (this->nextPipeline != nullptr) {
                        for (auto& newTask: tasks)
                            this->nextPipeline->feedTask(std::move(std::static_pointer_cast<ITask>(newTask)));
                    }
                }
            } catch (std::runtime_error e) {
                std::cerr << "Exception in pipeline: " << e.what() << std::endl;
            }
        }
    }

    void feedTask(std::shared_ptr<ITask> task) override { feedTaskImpl(std::move(task), true); }
    bool tryFeedTask(std::shared_ptr<ITask> task) override { return feedTaskImpl(std::move(task), false); }
protected:
    unsigned int maxTasks;
private:
    virtual std::vector<std::shared_ptr<ProducedTask>> work(std::shared_ptr<AcceptedTask>, unsigned int threadId) = 0;

    std::queue<std::shared_ptr<AcceptedTask>> queue;
    std::mutex queueLock;

    std::condition_variable consumerCv, producerCv;

    bool feedTaskImpl(std::shared_ptr<ITask> task, bool wait) {
        auto acceptedTask = std::dynamic_pointer_cast<AcceptedTask>(task);
        if (!acceptedTask) {
            throw std::runtime_error("Invalid task provided to pipeline");
        }

        std::unique_lock<std::mutex> lock(queueLock);
        if (queue.size() >= maxTasks) {
            if (wait)
                producerCv.wait(lock, [this]() { return this->queue.size() < maxTasks; });
            else return false;
        }

        queue.push(std::move(acceptedTask));
        lock.unlock();
        consumerCv.notify_all();

        return true;
    }
};

template<int num>
class Aggregator : public PipelineImpl<AggregatorAwareTask, AggregatedTask<num>> {
    // Not thread safe. Should always be executed in single thread.
    using MyPipelineImpl = PipelineImpl<AggregatorAwareTask, AggregatedTask<num>>;
public:
    using MyPipelineImpl::PipelineImpl;
    Aggregator(const Aggregator&) = delete;
    Aggregator& operator=(Aggregator) = delete;
    std::vector<std::shared_ptr<AggregatedTask<num>>> work(std::shared_ptr<AggregatorAwareTask> aggregatorAwareTask) override {
        if (taskMap.find(aggregatorAwareTask->taskId) == taskMap.end()) {
            taskMap[aggregatorAwareTask->taskId] = std::make_shared<AggregatedTask<num>>();
        }

        auto aggregatedTask = taskMap[aggregatorAwareTask->taskId];
        if (aggregatedTask->tasks[aggregatorAwareTask->aggregationId] != nullptr) {
            throw std::runtime_error("Duplicated aggregation task result");
        }

        aggregatedTask->tasks[aggregatorAwareTask->aggregationId] = aggregatorAwareTask;
        aggregatedTask->completedCount++;

        if (aggregatedTask->completedCount == num) {
            taskMap.erase(aggregatorAwareTask->taskId);
            return {aggregatedTask};
        } else {
            return {};
        }
    }
private:
    std::unordered_map<int, std::shared_ptr<AggregatedTask<num>>> taskMap;
};

#endif //EMBEDLIC_PIPELINE_H

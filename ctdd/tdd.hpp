#pragma once
#include "wnode.hpp"

namespace tdd {


	/// <summary>
	/// Note that methods in this class do not have validation check.
	/// </summary>
	/// <typeparam name="W"></typeparam>
	template <class W>
	class TDD {
	private:
		// The inner data of this tdd.
		node::weightednode<W> m_wnode;

		std::vector<int64_t> m_para_shape;

		// note that the extra inner index (2) is included in data_shape
		std::vector<int64_t> m_data_shape;

		// note: it is calculated from m_data_shape and m_index_order
		std::vector<int64_t> m_inner_data_shape;

		// The real index label of each inner data index.
		std::vector<int64_t> m_index_order;

		// The order of all indices it represents, including the extra inner dim (2).
		std::vector<int64_t> m_global_order;

	private:

		TDD(node::weightednode<W>&& w_node, std::vector<int64_t>&& para_shape,
			std::vector<int64_t>&& data_shape, std::vector<int64_t>&& index_order) {
			m_wnode = std::move(w_node);
			m_para_shape = std::move(para_shape);
			m_data_shape = std::move(data_shape);
			m_index_order = std::move(index_order);
			calculate_inner_data_shape();
			calculate_global_order();
		}

		void calculate_inner_data_shape() {
			
			auto temp = std::vector<int64_t>(m_index_order.size() + 1);
			temp[m_index_order.size()] = 2;
			for (int i = 0; i < m_index_order.size(); i++) {
				temp[i] = m_data_shape[m_index_order[i]];
			}
			m_inner_data_shape = std::move(temp);
		}

		void calculate_global_order() {
			auto dim_para = m_para_shape.size();
			auto dim_data = m_index_order.size();
			auto temp = std::vector<int64_t>(dim_para + dim_data + 1);
			temp[dim_para + dim_data] = dim_para + dim_data;
			for (int i = 0; i < dim_para; i++) {
				temp[i] = i;
			}
			for (int i = 0; i < dim_data; i++) {
				temp[i + dim_para] = m_index_order[i] + dim_para;
			}
			m_global_order = std::move(temp);
		}


	public:


		void reset() {
			cache::Global_Cache<W>::p_duplicate_cache->clear();
			cache::Global_Cache<W>::p_shift_cache->clear();
			cache::Global_Cache<W>::p_append_cache->clear();
			cache::Global_Cache<W>::p_CUDAcpl_cache->clear();
			cache::Global_Cache<W>::p_sum_cache->clear();
			cache::Global_Cache<W>::p_cont_cache->clear();
		}


		/// <summary>
		/// Construct a tdd tensor with the given direct representation.
		/// </summary>
		/// <param name="t">The given direct representation of a tensor.</param>
		/// <param name="dim_parallel">The number of parallel indices at front.</param>
		/// <param name="index_order">The index order used to stor this representation.
		/// Note that the item count + dim_parallel should be the dim of t strictly.
		/// If empty order is put in, the trival order will be taken.</param>
		/// <returns>The tdd created.</returns>
		static TDD<W> as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const std::vector<int64_t>& index_order) {
			auto dim_total = t.dim() - 1;
			auto dim_data = dim_total - dim_parallel;
			auto&& index_order_pd = std::vector<int64_t>(dim_data);
			if (index_order.empty()) {
				for (int i = 0; i < dim_data; i++) {
					index_order_pd[i] = i;
				}
			}
			else {
				for (int i = 0; i < dim_data; i++) {
					index_order_pd[i] = index_order[i];
				}
			}
			// prepare the data_parallel
			auto temp_para = std::vector<int64_t>(dim_parallel);
			for (int i = 0; i < dim_parallel; i++) {
				temp_para[i] = t.size(i);
			}
			// prepare the data_shape
			auto temp_data = std::vector<int64_t>(dim_data + 1);
			temp_data[dim_data] = 2;
			for (int i = 0; i < dim_data; i++) {
				temp_data[i] = t.size(i + dim_parallel);
			}
			auto w_node = wnode<W>::as_tensor_iterate(t, temp_para, temp_data, index_order_pd, 0);
			return TDD(std::move(w_node), std::move(temp_para), std::move(temp_data), std::move(index_order_pd));
		}



		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const {
			auto res = wnode<W>::to_CUDAcpl(m_wnode, m_inner_data_shape);

			// permute to the right index order
			res = res.permute(m_global_order);
			return res;
		}


		/// <summary>
		/// Sum up tdd a and b, and return the reduced result.
		/// This method will NOT check whether a and b are of the same shape.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		inline static TDD<W> sum(const TDD<W>& a, const TDD<W>& b) {
			auto res_wnode = wnode<W>::sum(a.m_wnode, b.m_wnode);
			return TDD(std::move(res_wnode),
				std::vector<int64_t>(a.m_para_shape),
				std::vector<int64_t>(a.m_data_shape),
				std::vector<int64_t>(a.m_index_order));
		}
	};
}
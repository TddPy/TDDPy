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

		// the range of each index it represents
		// note that the extra inner index (2) is included in data_shape
		std::vector<int64_t> m_data_shape;

		// note: it is calculated from m_data_shape and m_storage_order
		std::vector<int64_t> m_inner_data_shape;

		// The real index label of each inner data index.
		std::vector<int64_t> m_storage_order;

		// The "inversion" of index order of this tdd
		// (can be understood as the inverse in the permutation group)
		std::vector<int64_t> m_inversed_order;

		// The order of all indices it represents, including the extra inner dim (2).
		std::vector<int64_t> m_global_order;

		// The inverse of global order
		std::vector<int64_t> m_inversed_global_order;

	private:

		TDD(node::weightednode<W>&& w_node, std::vector<int64_t>&& para_shape,
			std::vector<int64_t>&& data_shape, std::vector<int64_t>&& storage_order) {
			m_wnode = std::move(w_node);
			m_para_shape = std::move(para_shape);
			m_data_shape = std::move(data_shape);
			m_storage_order = std::move(storage_order);
			calculate_inner_data_shape();
			calculate_inversed_order();
			calculate_global_order();
			calculate_inversed_global_order();
		}

		void calculate_inner_data_shape() {

			auto&& temp = std::vector<int64_t>(m_storage_order.size() + 1);
			temp[m_storage_order.size()] = 2;
			for (int i = 0; i < m_storage_order.size(); i++) {
				temp[i] = m_data_shape[m_storage_order[i]];
			}
			m_inner_data_shape = std::move(temp);
		}

		void calculate_inversed_order() {
			auto&& temp = std::vector<int64_t>(m_storage_order.size());
			for (int i = 0; i < m_storage_order.size(); i++) {
				temp[m_storage_order[i]] = i;
			}
			m_inversed_order = std::move(temp);
		}

		void calculate_global_order() {
			auto&& dim_para = m_para_shape.size();
			auto&& dim_data = m_storage_order.size();
			auto&& temp = std::vector<int64_t>(dim_para + dim_data + 1);
			temp[dim_para + dim_data] = dim_para + dim_data;
			for (int i = 0; i < dim_para; i++) {
				temp[i] = i;
			}
			for (int i = 0; i < dim_data; i++) {
				temp[i + dim_para] = m_storage_order[i] + dim_para;
			}
			m_global_order = std::move(temp);
		}

		void calculate_inversed_global_order() {
			auto&& dim_para = m_para_shape.size();
			auto&& dim_data = m_storage_order.size();
			auto&& temp = std::vector<int64_t>(dim_para + dim_data + 1);
			temp[dim_para + dim_data] = dim_para + dim_data;
			for (int i = 0; i < dim_para; i++) {
				temp[i] = i;
			}
			for (int i = 0; i < dim_data; i++) {
				temp[m_storage_order[i] + dim_para] = i + dim_para;
			}
			m_inversed_global_order = std::move(temp);
		}


		/// <summary>
		/// Return the data shape and storage_order after the reduction of specified indices.
		/// Note : Indices are counted in data indices only.
		/// </summary>
		/// <param name="indices_reduced">sorted. corresponds to inner data indices, not the indices of tensor it represents.</param>
		/// <returns>first: data shape, second: orders (not inner)</returns>
		std::pair<std::vector<int64_t>, std::vector<int64_t>> index_reduced_info(const std::vector<int64_t>& inner_indices_reduced) {
			auto&& length = inner_indices_reduced.size();
			std::vector<int64_t> orders(dim_data() - length);
			std::vector<int64_t> data_shapes(dim_data() + 1 - length);
			data_shapes[dim_data() - length] = 2;

			int new_count = 0;
			for (int i = 0; i < dim_data(); i++) {
				if (std::find(inner_indices_reduced.begin(), inner_indices_reduced.end(), i) == inner_indices_reduced.end()) {
					data_shapes[new_count] = m_inner_data_shape[i];
					orders[new_count] = m_storage_order[i];
					new_count++;
				}
			}

			std::vector<int64_t> index(orders.size());
			for (int i = 0; i < orders.size(); i++) {
				index[i] = i;
			}
			std::sort(index.begin(), index.end(),
				[orders](const int64_t& a, const int64_t& b) {
					return orders[a] < orders[b];
				});

			for (int i = 0; i < orders.size(); i++) {
				orders[index[i]] = i;
			}

			std::vector<int64_t> shapes_res(data_shapes);
			// sort the data_shape
			for (int i = 0; i < dim_data() - length; i++) {
				shapes_res[orders[i]] = data_shapes[i];
			}

			return std::make_pair(std::move(shapes_res), std::move(orders));
		}
	public:

		static void setting_update(bool device_cuda = false, double new_eps = DEFAULT_EPS) {
			CUDAcpl::reset(device_cuda);
			weight::EPS = new_eps;
		}

		static void reset() {
			node::Node<W>::reset();
			cache::Global_Cache<W>::p_CUDAcpl_cache->clear();
			cache::Global_Cache<W>::p_sum_cache->clear();
			cache::Global_Cache<W>::p_trace_cache->clear();
			cache::Global_Cache<W>::p_cont_cache->clear();
		}


		TDD(TDD&& other) {
			m_data_shape = std::move(other.m_data_shape);
			m_storage_order = std::move(other.m_storage_order);
			m_inner_data_shape = std::move(other.m_inner_data_shape);
			m_inversed_order = std::move(other.m_inversed_order);
			m_global_order = std::move(other.m_global_order);
			m_inversed_global_order = std::move(other.m_inversed_global_order);
			m_para_shape = std::move(other.m_para_shape);
			m_wnode = std::move(other.m_wnode);
		}

		TDD(const TDD& other) {
			m_data_shape = other.m_data_shape;
			m_storage_order = other.m_storage_order;
			m_inner_data_shape = other.m_inner_data_shape;
			m_inversed_order = other.m_inversed_order;
			m_global_order = other.m_global_order;
			m_inversed_global_order = other.m_inversed_global_order;
			m_para_shape = other.m_para_shape;
			m_wnode = other.m_wnode;
		}

		inline const node::weightednode<W>& w_node() const {
			return m_wnode;
		}

		inline const std::vector<int64_t>& parallel_shape()const {
			return m_para_shape;
		}

		inline int64_t dim_data()const {
			return m_storage_order.size();
		}

		inline const std::vector<int64_t>& data_shape() const {
			return m_data_shape;
		}

		inline const std::vector<int64_t>& storage_order() const {
			return m_storage_order;
		}

		inline int size() const {
			if (m_wnode.node == nullptr) {
				return 0;
			}
			else {
				return m_wnode.node->get_size();
			}
		}


		/// <summary>
		/// print the informaton of this tdd. DEBUG usage.
		/// </summary>
		inline void print() const {
			std::cout << "weight: " << m_wnode.weight << std::endl;
			std::cout << "node: " << m_wnode.node << std::endl;
			std::cout << "parallel shape: (" << m_para_shape << ")\n";
			std::cout << "data shape: (" << m_data_shape << ")\n";
			std::cout << "storage order: (" << m_storage_order << ")\n";
			std::cout << "inversed order: (" << m_inversed_order << ")\n";
			std::cout << "global order: (" << m_global_order << ")\n";
			std::cout << "inversed global order: (" << m_inversed_global_order << ")\n";
			//std::cout << "size: " << get_size() << std::endl;
		}

		inline void print_nodes() const {
			if (m_wnode.node != nullptr) {
				m_wnode.node->print();
			}
			else {
				std::cout << ">node: " << nullptr << std::endl;
			}
		}


		inline TDD<W> clone() const {
			return TDD(node::weightednode<W>(m_wnode),
				std::vector<int64_t>(m_para_shape),
				std::vector<int64_t>(m_data_shape),
				std::vector<int64_t>(m_storage_order));
		}

		/// <summary>
		/// Construct a tdd tensor with the given direct representation.
		/// </summary>
		/// <param name="t">The given direct representation of a tensor.</param>
		/// <param name="dim_parallel">The number of parallel indices at front.</param>
		/// <param name="storage_order">The index order used to stor this representation.
		/// Note that the item count + dim_parallel should be the dim of t strictly.
		/// If empty order is put in, the trival order will be taken.</param>
		/// <returns>The tdd created.</returns>
		static TDD<W> as_tensor(const CUDAcpl::Tensor& t, int dim_parallel, const std::vector<int64_t>& storage_order) {
			auto&& dim_total = t.dim() - 1;
			auto&& dim_data = dim_total - dim_parallel;
			auto&& storage_order_pd = std::vector<int64_t>(dim_data);
			if (storage_order.empty()) {
				for (int i = 0; i < dim_data; i++) {
					storage_order_pd[i] = i;
				}
			}
			else {
				for (int i = 0; i < dim_data; i++) {
					storage_order_pd[i] = storage_order[i];
				}
			}
			// prepare the data_parallel
			auto&& temp_para = std::vector<int64_t>(dim_parallel);
			for (int i = 0; i < dim_parallel; i++) {
				temp_para[i] = t.size(i);
			}
			// prepare the data_shape
			auto&& temp_data = std::vector<int64_t>(dim_data + 1);
			temp_data[dim_data] = 2;
			for (int i = 0; i < dim_data; i++) {
				temp_data[i] = t.size(i + dim_parallel);
			}
			auto&& w_node = wnode::as_tensor_iterate<W>(t, temp_para, temp_data, storage_order_pd, 0);
			return TDD(std::move(w_node), std::move(temp_para), std::move(temp_data), std::move(storage_order_pd));
		}



		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const {
			auto&& res = wnode::to_CUDAcpl(m_wnode, m_para_shape, m_inner_data_shape);
			// permute to the right index order
			res = res.permute(m_inversed_global_order);
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
			auto&& res_wnode = wnode::sum(a.m_wnode, b.m_wnode, a.m_para_shape);
			return TDD(std::move(res_wnode),
				std::vector<int64_t>(a.m_para_shape),
				std::vector<int64_t>(a.m_data_shape),
				std::vector<int64_t>(a.m_storage_order));
		}



		/// <summary>
		/// trace the tdd according to the specified data_indices. Return the reduced result.
		/// data_indices should be counted in the data indices only.
		/// </summary>
		/// <param name="indices">first less than second should hold for every pair</param>
		/// <returns></returns>
		TDD<W> trace(const cache::pair_cmd& indices) {
			if (indices.empty()) {
				return clone();
			}
			else {
				// transform to inner indices
				cache::pair_cmd inner_indices_cmd(indices.size());
				for (int i = 0; i < indices.size(); i++) {
					// arrange the smaller index at the first
					if (indices[i].first < indices[i].second) {
						inner_indices_cmd[i].first = m_inversed_order[indices[i].first];
						inner_indices_cmd[i].second = m_inversed_order[indices[i].second];
					}
					else {
						inner_indices_cmd[i].first = m_inversed_order[indices[i].second];
						inner_indices_cmd[i].second = m_inversed_order[indices[i].first];
					}
				}

				std::vector<int64_t> inner_i_reduced(indices.size() * 2);
				for (int i = 0; i < indices.size(); i++) {
					inner_i_reduced[2 * i] = inner_indices_cmd[i].first;
					inner_i_reduced[2 * i + 1] = inner_indices_cmd[i].second;
				}
				std::sort(inner_i_reduced.begin(), inner_i_reduced.end());

				//note that inner_indices.first < innier_indices.second should hold for every i
				auto&& res_wnode = wnode::trace(m_wnode, m_para_shape, m_inner_data_shape, inner_indices_cmd, inner_i_reduced);

				auto&& reduced_info = index_reduced_info(inner_i_reduced);

				return TDD(std::move(res_wnode), std::vector<int64_t>(m_para_shape),
					std::move(reduced_info.first), std::move(reduced_info.second));
			}
		}



		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="num_indices">contract the last num_indices indices of a and first of b</param>
		/// <param name="parallel_tensor"></param>
		/// <returns></returns>
		inline static TDD<W> tensordot_num(const TDD<W>& a, const TDD<W>& b, int num_indices, 
			const std::vector<int>& rearrangement = {}, bool parallel_tensor = false) {
			std::vector<int64_t> ia(num_indices);
			std::vector<int64_t> ib(num_indices);
			for (int i = 0; i < num_indices; i++) {
				ia[i] = a.dim_data() - num_indices + i;
				ib[i] = i;
			}
			return tensordot(a, b, ia, ib, rearrangement, parallel_tensor);
		}

		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// Whether to tensor on the parallel indices.
		/// </summary>
		/// <typeparam name="W"></typeparam>
		static TDD<W> tensordot(const TDD<W>& a, const TDD<W>& b,
			const std::vector<int64_t>& ils_a, const std::vector<int64_t>& ils_b, 
			const std::vector<int>& rearrangement = {}, bool parallel_tensor = false) {

			// transform to inner indices
			cache::pair_cmd inner_indices_cmd(ils_a.size());
			for (int i = 0; i < ils_a.size(); i++) {
				// arrange the smaller index at the first
				inner_indices_cmd[i].first = a.m_inversed_order[ils_a[i]];
				inner_indices_cmd[i].second = b.m_inversed_order[ils_b[i]];
			}

			std::vector<int> rearrangement_default;
			const std::vector<int>* p_rearrangement_pd;
			if (rearrangement.empty()) {
				auto length = a.dim_data() + b.dim_data() - 2 * ils_a.size();
				rearrangement_default = std::vector<int>(length);
				for (int i = 0; i < a.dim_data() - ils_a.size(); i++) {
					rearrangement_default[i] = 1;
				}
				for (int i = a.dim_data() - ils_a.size(); i < length; i++) {
					rearrangement_default[i] = 0;
				}
				p_rearrangement_pd = &rearrangement_default;
			}
			else {
				p_rearrangement_pd = &rearrangement;
			}

			// prepare the reduced inner indices
			std::vector<int64_t> a_reduced_indices(inner_indices_cmd.size());
			std::vector<int64_t> b_reduced_indices(inner_indices_cmd.size());
			int i = 0;
			for (const auto& cmd : inner_indices_cmd) {
				a_reduced_indices[i] = cmd.first;
				b_reduced_indices[i] = cmd.second;
				i++;
			}
			std::sort(a_reduced_indices.begin(), a_reduced_indices.end());
			std::sort(b_reduced_indices.begin(), b_reduced_indices.end());

			// prepare the order and shape (inner)
			std::vector<int64_t> total_order(p_rearrangement_pd->size());
			std::vector<int64_t> total_inner_shape(p_rearrangement_pd->size()+1);

			// the new inner order of each node in a and b
			std::vector<int64_t> a_inner_order(a.dim_data());
			std::vector<int64_t> b_inner_order(b.dim_data());

			auto&& i_red_a = a_reduced_indices.begin();
			auto&& i_red_b = b_reduced_indices.begin();
			int i_cur_a = 0;
			int i_cur_b = 0;

			// here i goes through the inner order
			for (int i = 0; i < p_rearrangement_pd->size(); i++) {
				if ((*p_rearrangement_pd)[i]) {
					// choice A
					while (i_red_a != a_reduced_indices.end()) {
						if (i_cur_a == *i_red_a) {
							i_red_a++;
							i_cur_a++;
						}
						else {
							break;
						}
					}
					total_order[i] = a.m_storage_order[i_cur_a];
					total_inner_shape[i] = a.m_inner_data_shape[i_cur_a];
					a_inner_order[i_cur_a] = i;
					i_cur_a++;
				}
				else {
					// choic B
					while (i_red_b != b_reduced_indices.end()) {
						if (i_cur_b == *i_red_b) {
							i_red_b++;
							i_cur_b++;
						}
						else {
							break;
						}
					}
					total_order[i] = b.m_storage_order[i_cur_b] + a.dim_data();
					total_inner_shape[i] = b.m_inner_data_shape[i_cur_b];
					b_inner_order[i_cur_b] = i;
					i_cur_b++;
				}
			}
			

			// transform to outer order
			std::vector<int64_t> index(total_order.size());
			for (int i = 0; i < total_order.size(); i++) {
				index[i] = i;
			}
			std::sort(index.begin(), index.end(),
				[total_order](const int64_t& a, const int64_t& b) {
					return total_order[a] < total_order[b];
				});
			for (int i = 0; i < total_order.size(); i++) {
				total_order[index[i]] = i;
			}

			std::vector<int64_t> total_shape(total_inner_shape.size());
			total_shape[total_inner_shape.size() - 1] = 2;
			// sort the data_shape
			for (int i = 0; i < total_inner_shape.size() - 1; i++) {
				total_shape[total_order[i]] = total_inner_shape[i];
			}

			// note that rearrangement does not need be processed.
			auto&& res_wnode = wnode::contract(a.m_wnode, b.m_wnode, a.m_para_shape, b.m_para_shape,
				a.m_inner_data_shape, b.m_inner_data_shape,
				inner_indices_cmd, a_inner_order, b_inner_order, parallel_tensor);

			std::vector<int64_t> new_para_shape(a.m_para_shape);
			if (parallel_tensor) {
				new_para_shape.insert(new_para_shape.end(), b.m_para_shape.begin(), b.m_para_shape.end());
			}

			return TDD(std::move(res_wnode), std::move(new_para_shape),
				std::move(total_shape), std::move(total_order));
		}

		/// <summary>
		/// permute the order of indices, and return the view.
		/// </summary>
		/// <param name="new_index_order"></param>
		/// <returns></returns>
		inline TDD<W> permute(const std::vector<int64_t>& permutation) {
			std::vector<int64_t> new_order(m_storage_order.size());
			std::vector<int64_t> new_data_shape(m_storage_order.size() + 1);
			new_data_shape[m_storage_order.size()] = 2;
			for (int i = 0; i < m_storage_order.size(); i++) {
				new_data_shape[i] = m_data_shape[permutation[i]];
				new_order[m_inversed_order[permutation[i]]] = i;
			}
			return TDD(node::weightednode<W>(m_wnode), std::vector<int64_t>(m_para_shape),
				std::move(new_data_shape), std::move(new_order));

		}
	};
}
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

		// The "inversion" of index order of this tdd
		// (can be understood as the inverse in the permutation group)
		std::vector<int64_t> m_inversed_order;

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
			calculate_inversed_order();
			calculate_global_order();
		}

		void calculate_inner_data_shape() {

			auto&& temp = std::vector<int64_t>(m_index_order.size() + 1);
			temp[m_index_order.size()] = 2;
			for (int i = 0; i < m_index_order.size(); i++) {
				temp[i] = m_data_shape[m_index_order[i]];
			}
			m_inner_data_shape = std::move(temp);
		}

		void calculate_inversed_order() {
			auto&& temp = std::vector<int64_t>(m_index_order.size());
			for (int i = 0; i < m_index_order.size(); i++) {
				temp[m_index_order[i]] = i;
			}
			m_inversed_order = std::move(temp);
		}

		void calculate_global_order() {
			auto&& dim_para = m_para_shape.size();
			auto&& dim_data = m_index_order.size();
			auto&& temp = std::vector<int64_t>(dim_para + dim_data + 1);
			temp[dim_para + dim_data] = dim_para + dim_data;
			for (int i = 0; i < dim_para; i++) {
				temp[i] = i;
			}
			for (int i = 0; i < dim_data; i++) {
				temp[i + dim_para] = m_index_order[i] + dim_para;
			}
			m_global_order = std::move(temp);
		}

		/// <summary>
		/// Return the data shape and index_order after the reduction of specified indices.
		/// Note : Indices are counted in data indices only.
		/// </summary>
		/// <param name="indices_reduced">corresponds to inner data indices, not the indices of tensor it represents.</param>
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
					orders[new_count] = m_index_order[i];
					new_count++;
				}
			}

			std::vector<int64_t> orders_res(dim_data() - length);
			for (int i = 0; i < dim_data() - length; i++) {
				orders_res[i] = i;
			}
			std::sort(orders_res.begin(), orders_res.end(),
				[orders](int64_t a, int64_t b) {
					return (orders[a] < orders[b]);
				});

			std::vector<int64_t> shapes_res(data_shapes);
			// sort the data_shape
			for (int i = 0; i < dim_data() - length; i++) {
				shapes_res[i] = data_shapes[orders_res[i]];
			}

			return std::make_pair(std::move(shapes_res), std::move(orders_res));
		}
	public:


		static void reset() {
			node::Node<W>::reset();
			cache::Global_Cache<W>::p_duplicate_cache->clear();
			cache::Global_Cache<W>::p_append_cache->clear();
			cache::Global_Cache<W>::p_CUDAcpl_cache->clear();
			cache::Global_Cache<W>::p_sum_cache->clear();
			cache::Global_Cache<W>::p_cont_cache->clear();
		}


		TDD(TDD&& other) {
			m_data_shape = std::move(other.m_data_shape);
			m_global_order = std::move(other.m_global_order);
			m_index_order = std::move(other.m_index_order);
			m_inner_data_shape = std::move(other.m_inner_data_shape);
			m_inversed_order = std::move(other.m_inversed_order);
			m_para_shape = std::move(other.m_para_shape);
			m_wnode = std::move(other.m_wnode);
		}

		TDD(const TDD& other) {
			m_data_shape = other.m_data_shape;
			m_global_order = other.m_global_order;
			m_index_order = other.m_index_order;
			m_inner_data_shape = other.m_inner_data_shape;
			m_inversed_order = other.m_inversed_order;
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
			return m_index_order.size();
		}

		inline const std::vector<int64_t>& data_shape() const {
			return m_data_shape;
		}

		inline const std::vector<int64_t>& index_order() const {
			return m_index_order;
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
			std::cout << "index order: (" << m_index_order << ")\n";
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
				std::vector<int64_t>(m_index_order));
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
			auto&& dim_total = t.dim() - 1;
			auto&& dim_data = dim_total - dim_parallel;
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
			auto&& w_node = wnode<W>::as_tensor_iterate(t, temp_para, temp_data, index_order_pd, 0);
			return TDD(std::move(w_node), std::move(temp_para), std::move(temp_data), std::move(index_order_pd));
		}



		/// <summary>
		/// Transform this tensor to a CUDA complex and return.
		/// </summary>
		/// <returns></returns>
		CUDAcpl::Tensor CUDAcpl() const {
			auto&& res = wnode<W>::to_CUDAcpl(m_wnode, m_inner_data_shape);

			// permute to the right index order
			res = res.permute(m_global_order);
			return res;
		}


		/// <summary>
		/// Return the direct product : a tensor b.The index order is the connection of that of a and b.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="parallel_tensor">
		/// whether to tensor on the parallel indices. False : parallel index of aand b must be the same, and their shapes are :
		///	a: [(? ), (s_a), 2] tensor b : [(? ), (s_b), 2] ->[(? ), (s_a), (s_b), 2]
		/// True : tensor on the parallel indices too.Their shapes are :
		/// a: [(? a), (s_a), 2] tensor b : [(? b), (s_b), 2] ->[(? a), (? b), (s_a), (s_b), 2]
		/// </param>
		/// <returns></returns>
		static TDD<W> direct_product(const TDD& a, const TDD& b, bool parallel_tensor = false) {
			if (parallel_tensor) {
				throw - 10;
			}
			auto&& w_node = wnode<W>::direct_product(a.m_wnode, a.dim_data(), b.m_wnode,
				a.m_para_shape, std::vector<int64_t>(), std::vector<int64_t>(), parallel_tensor);
			// adjust the data shape and index order
			auto&& new_index_order = std::vector<int64_t>(a.dim_data() + b.dim_data());
			for (int i = 0; i < a.dim_data(); i++) {
				new_index_order[i] = a.m_index_order[i];
			}
			for (int i = 0; i < b.dim_data(); i++) {
				new_index_order[i + a.dim_data()] = b.m_index_order[i] + a.dim_data();
			}

			auto&& new_data_shape = std::vector<int64_t>(a.dim_data() + b.dim_data() + 1);
			new_data_shape[a.dim_data() + b.dim_data()] = 2;
			for (int i = 0; i < a.dim_data(); i++) {
				new_data_shape[i] = a.m_data_shape[i];
			}
			for (int i = 0; i < b.dim_data(); i++) {
				new_data_shape[i + a.dim_data()] = b.m_data_shape[i];
			}
			auto&& new_para_shape = std::vector<int64_t>(a.m_para_shape);
			return TDD(std::move(w_node), std::move(new_para_shape), std::move(new_data_shape), std::move(new_index_order));
		}

		/// <summary>
		/// Sum up tdd a and b, and return the reduced result.
		/// This method will NOT check whether a and b are of the same shape.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		inline static TDD<W> sum(const TDD<W>& a, const TDD<W>& b) {
			auto&& res_wnode = wnode<W>::sum(a.m_wnode, b.m_wnode);
			return TDD(std::move(res_wnode),
				std::vector<int64_t>(a.m_para_shape),
				std::vector<int64_t>(a.m_data_shape),
				std::vector<int64_t>(a.m_index_order));
		}



		/// <summary>
		/// Contract the tdd according to the specified data_indices. Return the reduced result.
		/// data_indices should be counted in the data indices only.
		/// </summary>
		/// <param name="indices">first less than second should hold for every pair</param>
		/// <returns></returns>
		TDD<W> contract(const cache::cont_cmd& indices) {
			if (indices.empty()) {
				return clone();
			}
			else {
				// transform to inner indices
				cache::cont_cmd inner_indices_cmd(indices.size());
				for (int i = 0; i < indices.size(); i++) {
					inner_indices_cmd[i].first = m_inversed_order[indices[i].first];
					inner_indices_cmd[i].second = m_inversed_order[indices[i].second];

				}

				std::vector<int64_t> inner_i_reduced(indices.size() * 2);
				for (int i = 0; i < indices.size(); i++) {
					inner_i_reduced[2 * i] = inner_indices_cmd[i].first;
					inner_i_reduced[2 * i + 1] = inner_indices_cmd[i].second;
				}
				std::sort(inner_i_reduced.begin(), inner_i_reduced.end());

				//note that inner_indices.first < innier_indices.second should hold for every i
				auto&& res_wnode = wnode<W>::contract(m_wnode, m_inner_data_shape, inner_indices_cmd, inner_i_reduced);

				auto&& reduced_info = index_reduced_info(inner_i_reduced);

				return TDD(std::move(res_wnode), std::vector<int64_t>(m_para_shape),
					std::move(reduced_info.first), std::move(reduced_info.second));
			}
		}


		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// Whether to tensor on the parallel indices.
		/// </summary>
		/// <typeparam name="W"></typeparam>
		inline static TDD<W> tensordot(const TDD<W>& a, const TDD<W>& b,
			const std::vector<int64_t>& ils_a, const std::vector<int64_t>& ils_b, bool parallel_tensor = false) {

			auto&& temp_tdd = direct_product(a, b, parallel_tensor);

			cache::cont_cmd i_cmd(ils_a.size());

			for (int i = 0; i < ils_a.size(); i++) {
				i_cmd[i].first = ils_a[i];
				i_cmd[i].second = ils_b[i] + a.dim_data();
			}
			return temp_tdd.contract(i_cmd);
		}

		/// <summary>
		/// The pytorch-like tensordot method. Note that indices should be counted with data indices only.
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="num_indices">contract the last num_indices indices of a and first of b</param>
		/// <param name="parallel_tensor"></param>
		/// <returns></returns>
		inline static TDD<W> tensordot(const TDD<W>& a, const TDD<W>& b, int num_indices, bool parallel_tensor = false) {
			std::vector<int64_t> ia(num_indices);
			std::vector<int64_t> ib(num_indices);
			for (int i = 0; i < num_indices; i++) {
				ia[i] = a.dim_data() - num_indices + i;
				ib[i] = i;
			}
			return tensordot(a, b, ia, ib, parallel_tensor);
		}

		/// <summary>
		/// permute the order of indices, and return the view.
		/// </summary>
		/// <param name="new_index_order"></param>
		/// <returns></returns>
		inline TDD<W> permute(const std::vector<int64_t>& permutation) {
			std::vector<int64_t> new_order(m_index_order.size());
			std::vector<int64_t> new_data_shape(m_index_order.size() + 1);
			new_data_shape[m_index_order.size()] = 2;
			for (int i = 0; i < m_index_order.size(); i++) {
				new_data_shape[i] = m_data_shape[permutation[i]];
				new_order[i] = permutation[m_index_order[i]];
			}
			return TDD(node::weightednode<W>(m_wnode), std::vector<int64_t>(m_para_shape),
				std::move(new_data_shape), std::move(new_order));

		}
	};
}
#pragma once

#include "typing.hpp"
#include "wnode.hpp"
#include <xtensor/xarray.hpp>

namespace tdd {


	// Note that methods in this class do not have validation check.
	template <class T_WEIGHT>
	class TDD {
	private:
		wnode::WNode<T_WEIGHT> m_wnode;

		// the range of each index it represents
		std::vector<int64_t> m_data_shape;

		// The real index label of each inner data index.
		std::vector<int64_t> m_storage_order;

	private:

		TDD(wnode::WNode<T_WEIGHT>&& w_node,
			std::vector<int64_t>&& data_shape, std::vector<int64_t>&& storage_order) noexcept {
			m_wnode = std::move(w_node);
			m_data_shape = std::move(data_shape);
			m_storage_order = std::move(storage_order);
		}


    public:
        /// <summary>
		/// Construct a tdd tensor with the given matrix representation.
		/// </summary>
		/// <returns>The tdd created.</returns>
		static TDD<T_WEIGHT> as_tensor(Tensor& tensor, const std::vector<int64_t>& storage_order) {
			auto&& dim_total = tensor.dimension();

			auto&& storage_order_pd = std::vector<int64_t>(dim_total);
			if (storage_order.empty()) {
				for (int i = 0; i < dim_total; i++) {
					storage_order_pd[i] = i;
				}
			}
			else {
				for (int i = 0; i < dim_total; i++) {
					storage_order_pd[i] = storage_order[i];
				}
			}
			
			// prepare the data_shape
			// auto&& data_shape = tensor.shape();

			auto&& data_shape = std::vector<int64_t>(dim_total);
			// auto&& temp_data = std::vector<int64_t>(dim_total);
			// for (int i = 0; i < dim_total; i++) {
			// 	temp_data[i] = xt::shape tensor.shape(i);
			// }

			auto&& w_node = wnode::as_tensor_iterate<T_WEIGHT>(tensor, data_shape, storage_order_pd, 0);
			return TDD(std::move(w_node), std::move(data_shape), std::move(storage_order_pd));
		}

    };
}
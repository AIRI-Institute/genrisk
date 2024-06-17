def plot_worst_old(ans):
    test_ts_values = np.array(list(test_ts.values()))
    ans_values = np.array(list(ans.values()))

    error_val = (ans_values - test_ts_values) ** 2
    error0 = error_val.reshape(-1)

    shift_model = ConditionalShift(
        mutable_columns=list(test_df.columns),
        immutable_columns=list(test_cov_df.columns),
        alpha_space=np.linspace(0.05, 0.95, 10),
        cv=5,
        # mode = 'gbr_quantiles'
    )

    risk0 = shift_model.fit(result_df, error0)

    errors = []
    for fake in fake_data:
        errors += (
            ((ans_values.reshape(-1) - np.array(fake[col].values)) ** 2)
            .reshape(-1)
            .tolist()
        )

    errors += error0.tolist()
    error_val = np.array(errors)
    fake_df = pd.concat(
        [fake_data[0], fake_data[1], fake_data[2], result_df], ignore_index=True
    )
    risk_fake = shift_model.fit(fake_df, error_val)

    ####
    risk0 = np.array(risk0)
    risk_fake = np.array(risk_fake)

    alpha_space = np.linspace(0.05, 0.95, 10)

    plt.plot(alpha_space, risk0[:, 0], label="test 1")
    plt.fill_between(alpha_space, risk0[:, 1], risk0[:, 2], alpha=0.1)

    plt.plot(alpha_space, risk_fake[:, 0], label="test 2")
    plt.fill_between(alpha_space, risk_fake[:, 1], risk_fake[:, 2], alpha=0.1)

    plt.legend()
    plt.show()

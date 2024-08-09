import streamlit as st


def intro():
    import streamlit as st

    st.write("# Welcome to DeepPrep! 👋")
    st.sidebar.success("Select a config above.")

    st.markdown(
        """
        DeepPrep: An accelerated, scalable, and robust pipeline for 
        neuroimaging preprocessing empowered by deep learning.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what DeepPrep can do!

        ### Want to learn more?

        - Check out [DeepPrep document](https://deepprep.readthedocs.io/en/latest/)
        - Ask a question in our [issues page](https://github.com/pBFSLab/DeepPrep)

        ### Citation  
        Ren, J., An, N., Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. and Wang, W., 2024. *DeepPrep: An accelerated, scalable, and robust pipeline for neuroimaging preprocessing empowered by deep learning*. bioRxiv, pp.2024-03.  
        
        ### License  
        Copyright 2023 The DeepPrep Developers  
        
        Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at  
        
           [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)  
        
        Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. 

    """
    )


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


page_names_to_funcs = {
    "—": intro,
    "preprocess anat & BOLD": plotting_demo,
    "preprocess anat only": plotting_demo,
    "preprocess BOLD only": plotting_demo
}

demo_name = st.sidebar.selectbox("Choose a config", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

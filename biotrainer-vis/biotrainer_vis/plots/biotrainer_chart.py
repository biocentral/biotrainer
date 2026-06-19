import altair as alt

from typing import List
from biotrainer_core.data_classes import SequenceData

from .plotting import plot_label_distribution


class BiotrainerChart:
    def __init__(self, chart: alt.Chart):
        self.chart = chart

    @classmethod
    def label_distribution(cls, dataset: List[SequenceData]):
        chart = plot_label_distribution(dataset)
        return cls(chart)

    def export(self, output_path: str):
        raise NotImplementedError
        """
        # Export
    svg_path = f'{output_prefix}.svg'
    metadata_path = f'{output_prefix}_metadata.json'

    result = export_altair_chart(chart, svg_path, metadata_path)

    # Enrich metadata with biological context
    points = result['metadata'].get('points', [])

    # Match points to data (by order, since Altair maintains order)
    for i, point in enumerate(points):
        if i < len(df):
            row = df.iloc[i]
            label = row['label']

            # Add enriched data
            point['data'].update({
                'label': label,
                'count': int(row['count']),
                'percentage': float(row['percentage']),
                'avg_length': float(row['avg_length']),
                'sets': data_by_label[label]['sets'],
                'sample_ids': data_by_label[label]['seq_ids'][:5],  # First 5 IDs
            })

    # Save enriched metadata
    import json
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Enhanced metadata saved with {len(points)} bars")
    print(f"📊 Labels: {list(data_by_label.keys())}")

    return result
        """

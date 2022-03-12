class CreateRecyclerCategories < ActiveRecord::Migration[7.0]
  def change
    create_table :recycler_categories do |t|
      t.references :recycler, null: false, foreign_key: true
      t.references :category, null: false, foreign_key: true

      t.timestamps
    end
  end
end

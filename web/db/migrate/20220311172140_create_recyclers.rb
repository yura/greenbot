class CreateRecyclers < ActiveRecord::Migration[7.0]
  def change
    create_table :recyclers do |t|
      t.string :name
      t.text :description
      t.string :phone
      t.string :email
      t.references :city, null: false, foreign_key: true
      t.text :address
      t.string :url

      t.timestamps
    end
  end
end

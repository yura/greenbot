class Recycler < ApplicationRecord
  belongs_to :city
  has_many :recycler_categories
  has_many :categories, through: :recycler_categories
end
